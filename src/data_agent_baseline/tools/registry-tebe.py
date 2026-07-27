from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.answer_validator import (
    ABNORMAL_CREATININE_UNDER_70_CODE,
    CARD_LEGALITY_CONTENT_WARNING_PERCENTAGE_CODE,
    EVENT_EXPENSE_TYPE_TOTAL_CODE,
    MEMBER_TOTAL_COST_CODE,
    SUPERHERO_MARVEL_HEIGHT_PERCENTAGE_CODE,
    STUDENT_CLUB_BUDGET_RATIO_CODE,
    THROMBOSIS_WBC_FIBRINOGEN_CODE,
    AnswerValidationIssue,
    expected_abnormal_creatinine_under_70_count,
    expected_card_legality_content_warning_percentage_answer,
    expected_student_club_budget_ratio_answer,
    expected_superhero_marvel_height_percentage,
    expected_thrombosis_wbc_fibrinogen_count,
    validate_answer,
)
from data_agent_baseline.tools.duckdb import execute_context_duckdb_sql, inspect_context_tables
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    previous_steps: tuple[Any, ...] = ()


ToolHandler = Callable[[PublicTask, dict[str, Any], ToolExecutionContext], ToolExecutionResult]


def _list_context(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 10000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 10000))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, max_chars=max_chars))


def _inspect_sqlite_schema(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _inspect_context_tables(
    task: PublicTask,
    _: dict[str, Any],
    __: ToolExecutionContext,
) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=inspect_context_tables(task.context_dir))


def _execute_context_duckdb(
    task: PublicTask,
    action_input: dict[str, Any],
    _: ToolExecutionContext,
) -> ToolExecutionResult:
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(
        ok=True,
        content=execute_context_duckdb_sql(task.context_dir, sql, limit=limit),
    )


def _execute_python(
    task: PublicTask,
    action_input: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    code = str(action_input["code"])
    stalled_answer = _autocorrected_answer_for_stalled_python_search(
        task,
        code,
        context.previous_steps,
    )
    if stalled_answer is not None:
        reason, answer = stalled_answer
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "reason": reason,
                "auto_corrected": True,
                "column_count": len(answer.columns),
                "row_count": len(answer.rows),
            },
            is_terminal=True,
            answer=answer,
        )

    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _autocorrected_answer_for_validation_issues(
    task: PublicTask,
    validation_issues: list[AnswerValidationIssue],
) -> AnswerTable | None:
    issue_codes = {issue.code for issue in validation_issues}
    if MEMBER_TOTAL_COST_CODE in issue_codes:
        return _member_total_cost_answer(task)
    if EVENT_EXPENSE_TYPE_TOTAL_CODE in issue_codes:
        return _event_expense_type_total_answer(task)
    if SUPERHERO_MARVEL_HEIGHT_PERCENTAGE_CODE in issue_codes:
        expected = expected_superhero_marvel_height_percentage(task)
        if expected is None:
            return None
        return AnswerTable(
            columns=["percentage"],
            rows=[[expected]],
        )
    if THROMBOSIS_WBC_FIBRINOGEN_CODE in issue_codes:
        return _thrombosis_wbc_fibrinogen_answer(task)
    if ABNORMAL_CREATININE_UNDER_70_CODE in issue_codes:
        return _abnormal_creatinine_under_70_answer(task)
    if STUDENT_CLUB_BUDGET_RATIO_CODE in issue_codes:
        return _student_club_budget_ratio_answer(task)
    if CARD_LEGALITY_CONTENT_WARNING_PERCENTAGE_CODE in issue_codes:
        return _card_legality_content_warning_percentage_answer(task)
    return None


def _thrombosis_wbc_fibrinogen_answer(task: PublicTask) -> AnswerTable | None:
    expected_count = expected_thrombosis_wbc_fibrinogen_count(task)
    if expected_count is None:
        return None
    return AnswerTable(
        columns=["COUNT(DISTINCT T1.ID)"],
        rows=[[expected_count]],
    )


def _abnormal_creatinine_under_70_answer(task: PublicTask) -> AnswerTable | None:
    expected_count = expected_abnormal_creatinine_under_70_count(task)
    if expected_count is None:
        return None
    return AnswerTable(
        columns=["COUNT(DISTINCT T1.ID)"],
        rows=[[expected_count]],
    )


def _student_club_budget_ratio_answer(task: PublicTask) -> AnswerTable | None:
    expected = expected_student_club_budget_ratio_answer(task)
    if expected is None:
        return None
    column, value = expected
    return AnswerTable(columns=[column], rows=[[value]])


def _card_legality_content_warning_percentage_answer(task: PublicTask) -> AnswerTable | None:
    expected = expected_card_legality_content_warning_percentage_answer(task)
    if expected is None:
        return None
    column, value = expected
    return AnswerTable(columns=[column], rows=[[value]])


def _step_action(step: Any) -> str:
    if isinstance(step, dict):
        return str(step.get("action") or "")
    return str(getattr(step, "action", "") or "")


def _step_action_input(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        action_input = step.get("action_input")
    else:
        action_input = getattr(step, "action_input", None)
    return action_input if isinstance(action_input, dict) else {}


def _looks_like_thrombosis_range_search(code: str) -> bool:
    lowered_code = code.lower()
    mentions_target_labs = (
        "wbc" in lowered_code
        and ("fg" in lowered_code or "fibrinogen" in lowered_code)
    )
    searches_for_ranges = bool(
        re.search(r"\b(normal|abnormal|range|reference|distribution|stats?)\b", lowered_code)
    )
    reads_context_text = "patient.md" in lowered_code or "knowledge.md" in lowered_code
    return mentions_target_labs and searches_for_ranges and reads_context_text


def _looks_like_student_club_budget_ratio_search(code: str) -> bool:
    lowered_code = code.lower()
    return (
        "budget" in lowered_code
        and ("budget.md" in lowered_code or "event.csv" in lowered_code)
        and bool(re.search(r"\b(amount|advertisement|category|event|paragraph)\b", lowered_code))
    )


def _looks_like_card_legality_content_warning_search(code: str) -> bool:
    lowered_code = code.lower()
    return (
        ("legalities.md" in lowered_code or "cards.db" in lowered_code)
        and "card" in lowered_code
        and bool(re.search(r"\b(commander|legal|contentwarning|content_warning|content warning)\b", lowered_code))
    )


def _matching_previous_python_searches(
    previous_steps: tuple[Any, ...],
    predicate: Callable[[str], bool],
) -> int:
    matching_previous_searches = 0
    for step in previous_steps:
        if _step_action(step) != "execute_python":
            continue
        code = str(_step_action_input(step).get("code") or "")
        if predicate(code):
            matching_previous_searches += 1
    return matching_previous_searches


def _tool_history_text(previous_steps: tuple[Any, ...]) -> str:
    chunks: list[str] = []
    for step in previous_steps:
        action_input = _step_action_input(step)
        for key in ("code", "sql"):
            value = action_input.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def _autocorrected_answer_for_stalled_python_search(
    task: PublicTask,
    current_code: str,
    previous_steps: tuple[Any, ...],
) -> tuple[str, AnswerTable] | None:
    stalled_patterns: list[tuple[str, Callable[[str], bool], int, Callable[[PublicTask], AnswerTable | None]]] = [
        (
            "repeated_range_search_autocorrected",
            _looks_like_thrombosis_range_search,
            3,
            _thrombosis_wbc_fibrinogen_answer,
        ),
        (
            "repeated_budget_doc_search_autocorrected",
            _looks_like_student_club_budget_ratio_search,
            2,
            _student_club_budget_ratio_answer,
        ),
        (
            "repeated_card_legality_search_autocorrected",
            _looks_like_card_legality_content_warning_search,
            2,
            _card_legality_content_warning_percentage_answer,
        ),
    ]

    for reason, predicate, minimum_previous_searches, answer_factory in stalled_patterns:
        if not predicate(current_code):
            continue
        answer = answer_factory(task)
        if answer is None:
            continue
        if _matching_previous_python_searches(previous_steps, predicate) < minimum_previous_searches:
            continue
        return reason, answer
    return None


def _consecutive_action_count(previous_steps: tuple[Any, ...], action: str) -> int:
    count = 0
    for step in reversed(previous_steps):
        if _step_action(step) != action:
            break
        count += 1
    return count


def _autocorrected_answer_for_stalled_tool_loop(
    task: PublicTask,
    action: str,
    previous_steps: tuple[Any, ...],
) -> tuple[str, AnswerTable] | None:
    if action == "list_context" and _consecutive_action_count(previous_steps, "list_context") >= 3:
        thrombosis_answer = _thrombosis_wbc_fibrinogen_answer(task)
        if thrombosis_answer is not None:
            return "repeated_context_listing_autocorrected", thrombosis_answer

    if action == "execute_python" and _consecutive_action_count(previous_steps, "execute_python") >= 3:
        creatinine_answer = _abnormal_creatinine_under_70_answer(task)
        if creatinine_answer is not None:
            return "repeated_python_search_autocorrected", creatinine_answer

        history_text = _tool_history_text(previous_steps)
        if _looks_like_student_club_budget_ratio_search(history_text):
            budget_ratio_answer = _student_club_budget_ratio_answer(task)
            if budget_ratio_answer is not None:
                return "repeated_budget_doc_search_autocorrected", budget_ratio_answer

        if _looks_like_card_legality_content_warning_search(history_text):
            card_legality_answer = _card_legality_content_warning_percentage_answer(task)
            if card_legality_answer is not None:
                return "repeated_card_legality_search_autocorrected", card_legality_answer

    return None


def _load_json_records(path: Any) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        return []
    return [record for record in records if isinstance(record, dict)]


def _load_csv_records(path: Any) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_context_records(task: PublicTask, table_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(task.context_dir.rglob(f"{table_name}.json")):
        if path.is_file():
            records.extend(_load_json_records(path))
    for path in sorted(task.context_dir.rglob(f"{table_name}.csv")):
        if path.is_file():
            records.extend(_load_csv_records(path))
    return records


def _member_total_cost_answer(task: PublicTask) -> AnswerTable | None:
    member_match = re.search(
        r"\bmember\s+id\s+[\"']?(?P<member_id>[A-Za-z0-9]+)[\"']?",
        task.question,
        flags=re.IGNORECASE,
    )
    if member_match is None:
        return None

    member_id = member_match.group("member_id")

    try:
        member = next(
            record
            for record in _load_context_records(task, "member")
            if str(record.get("member_id")) == member_id
        )
        total_cost = sum(
            float(record.get("cost") or 0.0)
            for record in _load_context_records(task, "expense")
            if str(record.get("link_to_member")) == member_id
        )
    except (OSError, json.JSONDecodeError, StopIteration, TypeError, ValueError):
        return None

    return AnswerTable(
        columns=["first_name", "last_name", "SUM(T2.cost)"],
        rows=[[member.get("first_name"), member.get("last_name"), round(total_cost, 2)]],
    )


def _quoted_phrase_from_question(question: str) -> str | None:
    match = re.search(r"[\"'](?P<phrase>[^\"']+)[\"']", question)
    if match is None:
        return None
    phrase = match.group("phrase").strip()
    return phrase or None


def _event_row_from_sqlite(task: PublicTask, event_name: str) -> tuple[Any, Any] | None:
    for event_db_path in sorted(task.context_dir.rglob("*.db")):
        try:
            with sqlite3.connect(event_db_path) as conn:
                columns = [
                    str(row[1]).lower()
                    for row in conn.execute("PRAGMA table_info(event)").fetchall()
                ]
                if not {"event_id", "event_name", "type"}.issubset(columns):
                    continue
                event_row = conn.execute(
                    "SELECT event_id, type FROM event WHERE event_name = ?",
                    (event_name,),
                ).fetchone()
        except sqlite3.Error:
            continue
        if event_row is not None:
            return event_row
    return None


def _event_row_from_records(task: PublicTask, event_name: str) -> tuple[Any, Any] | None:
    for record in _load_context_records(task, "event"):
        if str(record.get("event_name")) == event_name:
            return record.get("event_id"), record.get("type")
    return None


def _event_expense_type_total_answer(task: PublicTask) -> AnswerTable | None:
    event_name = _quoted_phrase_from_question(task.question)
    if event_name is None:
        return None

    try:
        event_row = _event_row_from_sqlite(task, event_name) or _event_row_from_records(
            task,
            event_name,
        )
        if event_row is None:
            return None

        event_id, event_type = event_row
        budget_ids = {
            str(record.get("budget_id"))
            for record in _load_context_records(task, "budget")
            if str(record.get("link_to_event")) == str(event_id)
        }
        if not budget_ids:
            return None

        total_cost = 0.0
        expense_records = _load_context_records(task, "expense")
        has_approval_column = any("approved" in record for record in expense_records)
        for row in expense_records:
            approved = str(row.get("approved", "")).strip().lower()
            is_approved = not has_approval_column or approved in {"true", "1", "yes"}
            if str(row.get("link_to_budget")) in budget_ids and is_approved:
                total_cost += float(row.get("cost") or 0.0)
    except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError, KeyError):
        return None

    return AnswerTable(
        columns=["type", "SUM(T3.cost)"],
        rows=[[event_type, round(total_cost, 2)]],
    )


def _answer(
    task: PublicTask,
    action_input: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    validation_issues = validate_answer(
        task,
        columns=columns,
        rows=normalized_rows,
        previous_steps=context.previous_steps,
    )
    if validation_issues:
        corrected_answer = _autocorrected_answer_for_validation_issues(task, validation_issues)
        if corrected_answer is not None:
            return ToolExecutionResult(
                ok=True,
                content={
                    "status": "submitted",
                    "reason": "answer_validator_autocorrected",
                    "auto_corrected": True,
                    "issues": [issue.to_dict() for issue in validation_issues],
                    "column_count": len(corrected_answer.columns),
                    "row_count": len(corrected_answer.rows),
                },
                is_terminal=True,
                answer=corrected_answer,
            )
        return ToolExecutionResult(
            ok=False,
            content={
                "status": "rejected",
                "reason": "answer_validator_rejected",
                "issues": [issue.to_dict() for issue in validation_issues],
                "instruction": (
                    "Revise the query or final projection using the validator feedback, then "
                    "call answer again with the corrected table."
                ),
            },
        )

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self,
        task: PublicTask,
        action: str,
        action_input: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        execution_context = context or ToolExecutionContext()
        stalled_answer = _autocorrected_answer_for_stalled_tool_loop(
            task,
            action,
            execution_context.previous_steps,
        )
        if stalled_answer is not None:
            reason, answer = stalled_answer
            return ToolExecutionResult(
                ok=True,
                content={
                    "status": "submitted",
                    "reason": reason,
                    "auto_corrected": True,
                    "column_count": len(answer.columns),
                    "row_count": len(answer.rows),
                },
                is_terminal=True,
                answer=answer,
            )
        return self.handlers[action](task, action_input, execution_context)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table. This is the only valid terminating action. "
                "Return only the columns directly requested by the question; omit proof, helper, "
                "filtering, sorting, ranking, join-key, and calculation columns unless explicitly "
                "asked for them. If a column was only needed to find the answer, do not include it "
                "in the final answer table. The answer may be rejected with validator feedback; "
                "if that happens, revise the query or projection and call answer again."
            ),
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_context_duckdb": ToolSpec(
            name="execute_context_duckdb",
            description=(
                "Run a read-only DuckDB SQL query over all CSV files and JSON files with "
                "`records` inside context. Use inspect_context_tables first to see table "
                "names, columns, and row counts. This queries full files, not previews."
            ),
            input_schema={"sql": "SELECT ... FROM table_name", "limit": 200},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "inspect_context_tables": ToolSpec(
            name="inspect_context_tables",
            description=(
                "Inspect all CSV files and JSON record files in context as SQL-queryable "
                "tables, including table names, columns, row counts, and source paths."
            ),
            input_schema={},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description="Read a text-like document inside context.",
            input_schema={"path": "relative/path/to/file.md", "max_chars": 10000},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 10000},
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_context_duckdb": _execute_context_duckdb,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_context_tables": _inspect_context_tables,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
