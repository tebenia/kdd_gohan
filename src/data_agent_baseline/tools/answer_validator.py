from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.duckdb import inspect_context_tables


HELPER_COLUMN_NAMES = {
    "account_id",
    "amount",
    "average",
    "avg",
    "balance",
    "cost",
    "date",
    "maximum",
    "maximum_cost",
    "minimum",
    "minimum_cost",
    "operation",
    "proof",
    "rank",
    "ranking",
    "sum",
    "total",
    "total_cost",
    "type",
}

MINMAX_PATTERN = re.compile(
    r"\b(lowest|highest|minimum|maximum|min|max|least|most|smallest|largest|"
    r"cheapest|costliest|earliest|latest)\b",
    flags=re.IGNORECASE,
)

EXPLICIT_ONE_PATTERN = re.compile(
    r"\b(exactly one|single|first|nearest|closest|top 1|one result|one row)\b",
    flags=re.IGNORECASE,
)

EXPLICIT_EXTRA_DETAIL_PATTERN = re.compile(
    r"\b(include|including|along with|together with|with its|with their|and its|and their)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnswerValidationIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def validate_answer(
    task: PublicTask,
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    previous_steps: Sequence[Any] = (),
) -> list[AnswerValidationIssue]:
    """Return conservative, actionable issues for suspicious final answers.

    The validator intentionally checks only high-confidence patterns. A false reject costs an
    agent step, so broad semantic judging still belongs in the model prompt and tools.
    """

    schema_tables = _load_schema_tables(task)
    recent_steps = _steps_since_last_rejected_answer(previous_steps)

    issues: list[AnswerValidationIssue] = []
    issues.extend(_extra_column_issues(task.question, columns))
    issues.extend(_merged_name_issues(columns, schema_tables))
    issues.extend(_minmax_limit_issues(task.question, recent_steps))
    issues.extend(_entity_attribute_issues(task.question, columns, schema_tables, recent_steps))
    issues.extend(_ambiguous_same_name_issues(columns, schema_tables, recent_steps))
    return _deduplicate_issues(issues)


def _load_schema_tables(task: PublicTask) -> list[dict[str, Any]]:
    try:
        payload = inspect_context_tables(task.context_dir)
    except Exception:
        return []
    tables = payload.get("tables", [])
    if not isinstance(tables, list):
        return []
    return [table for table in tables if isinstance(table, dict)]


def _deduplicate_issues(issues: list[AnswerValidationIssue]) -> list[AnswerValidationIssue]:
    seen: set[str] = set()
    deduped: list[AnswerValidationIssue] = []
    for issue in issues:
        key = f"{issue.code}:{issue.message}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _question_has_minmax(question: str) -> bool:
    return bool(MINMAX_PATTERN.search(question))


def _question_allows_single_minmax(question: str) -> bool:
    return bool(EXPLICIT_ONE_PATTERN.search(question))


def _question_requests_extra_details(question: str) -> bool:
    return bool(EXPLICIT_EXTRA_DETAIL_PATTERN.search(question))


def _is_helper_column(column: str) -> bool:
    normalized = _normalize_identifier(column)
    return normalized in HELPER_COLUMN_NAMES or normalized.endswith("_proof")


def _extra_column_issues(question: str, columns: Sequence[str]) -> list[AnswerValidationIssue]:
    if len(columns) <= 1 or _question_requests_extra_details(question):
        return []

    helper_columns = [column for column in columns if _is_helper_column(column)]
    if not helper_columns:
        return []

    lowered_question = question.lower()
    is_which_minmax = "which" in lowered_question and _question_has_minmax(question)
    is_list_all = bool(re.search(r"\b(list|show|return)\s+all\b", lowered_question))
    if not is_which_minmax and not is_list_all:
        return []

    return [
        AnswerValidationIssue(
            code="extra_helper_columns",
            message=(
                "The answer includes likely helper/proof columns "
                f"{helper_columns}. Return only the fields directly requested by the question; "
                "use helper columns only internally unless explicitly requested."
            ),
        )
    ]


def _table_columns(table: dict[str, Any]) -> list[str]:
    columns = table.get("columns", [])
    if not isinstance(columns, list):
        return []
    return [str(column) for column in columns]


def _table_name(table: dict[str, Any]) -> str:
    return str(table.get("table") or "")


def _column_table_map(schema_tables: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    column_tables: dict[str, list[str]] = {}
    for table in schema_tables:
        table_name = _normalize_identifier(_table_name(table))
        for column in _table_columns(table):
            column_tables.setdefault(_normalize_identifier(column), []).append(table_name)
    return column_tables


def _merged_name_issues(
    columns: Sequence[str],
    schema_tables: Sequence[dict[str, Any]],
) -> list[AnswerValidationIssue]:
    answer_columns = {_normalize_identifier(column) for column in columns}
    if not answer_columns.intersection({"full_name", "fullname"}):
        return []

    for table in schema_tables:
        table_columns = {_normalize_identifier(column) for column in _table_columns(table)}
        has_split_name = {"first_name", "last_name"}.issubset(table_columns) or {
            "first",
            "last",
        }.issubset(table_columns)
        if has_split_name:
            return [
                AnswerValidationIssue(
                    code="merged_name_columns",
                    message=(
                        "The answer uses a merged full_name column, but the source schema has "
                        "separate first/last name columns. Preserve source granularity and return "
                        "the split name columns unless the task explicitly requires a merged field."
                    ),
                )
            ]
    return []


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


def _step_observation(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        observation = step.get("observation")
    else:
        observation = getattr(step, "observation", None)
    return observation if isinstance(observation, dict) else {}


def _steps_since_last_rejected_answer(previous_steps: Sequence[Any]) -> list[Any]:
    start_index = 0
    for index, step in enumerate(previous_steps):
        if _step_action(step) != "answer":
            continue
        observation = _step_observation(step)
        if observation.get("ok") is False:
            start_index = index + 1
    return list(previous_steps[start_index:])


def _last_sql_text(steps: Sequence[Any]) -> str | None:
    for step in reversed(steps):
        if _step_action(step) not in {"execute_context_duckdb", "execute_context_sql"}:
            continue
        sql = _step_action_input(step).get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
    return None


def _minmax_limit_issues(question: str, steps: Sequence[Any]) -> list[AnswerValidationIssue]:
    if not _question_has_minmax(question) or _question_allows_single_minmax(question):
        return []

    sql = _last_sql_text(steps)
    if sql is None or not re.search(r"\blimit\s+1\b", sql, flags=re.IGNORECASE):
        return []

    return [
        AnswerValidationIssue(
            code="minmax_limit_one",
            message=(
                "The latest SQL query uses LIMIT 1 for a lowest/highest/min/max question. "
                "This can miss tied rows. Recompute the min/max value, select every row equal "
                "to that value, and then submit only the requested output columns."
            ),
        )
    ]


def _recent_history_text(steps: Sequence[Any]) -> str:
    chunks: list[str] = []
    for step in steps:
        chunks.append(_step_action(step))
        action_input = _step_action_input(step)
        if action_input:
            chunks.append(json.dumps(action_input, ensure_ascii=False, sort_keys=True))
        if isinstance(step, dict):
            raw_response = step.get("raw_response")
        else:
            raw_response = getattr(step, "raw_response", None)
        if isinstance(raw_response, str):
            chunks.append(raw_response)
    return "\n".join(chunks).lower()


def _entity_attribute_issues(
    question: str,
    columns: Sequence[str],
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    answer_columns = {_normalize_identifier(column) for column in columns}
    column_tables = _column_table_map(schema_tables)

    asks_driver_number = (
        "driver" in lowered_question
        and "number" in lowered_question
        and "number" in answer_columns
        and "drivers" in column_tables.get("number", [])
        and len(set(column_tables.get("number", []))) > 1
    )
    if not asks_driver_number:
        return []

    history_text = _recent_history_text(steps)
    if re.search(r"\bdrivers\b", history_text):
        return []

    return [
        AnswerValidationIssue(
            code="driver_number_requires_drivers_table",
            message=(
                "The question asks for the number of the driver, and the context has both "
                "`drivers.number` and another table's `number`. Use the event table only to "
                "identify matching driverId rows, then join/read `drivers` and return "
                "`drivers.number`."
            ),
        )
    ]


def _referenced_tables(sql: str, schema_tables: Sequence[dict[str, Any]]) -> set[str]:
    lowered_sql = sql.lower()
    referenced: set[str] = set()
    for table in schema_tables:
        table_name = _normalize_identifier(_table_name(table))
        if table_name and re.search(rf"(?<![\w]){re.escape(table_name)}(?![\w])", lowered_sql):
            referenced.add(table_name)
    return referenced


def _selects_unqualified_column(sql: str, column: str) -> bool:
    select_match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if select_match is None:
        return False

    select_clause = select_match.group(1)
    normalized_column = _normalize_identifier(column)
    bare_column_pattern = re.compile(
        rf"(?<![.\w])\"?{re.escape(normalized_column)}\"?(?![\w])",
        flags=re.IGNORECASE,
    )
    return bool(bare_column_pattern.search(select_clause))


def _ambiguous_same_name_issues(
    columns: Sequence[str],
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    sql = _last_sql_text(steps)
    if sql is None:
        return []

    column_tables = _column_table_map(schema_tables)
    referenced_tables = _referenced_tables(sql, schema_tables)
    issues: list[AnswerValidationIssue] = []

    for column in columns:
        normalized_column = _normalize_identifier(column)
        tables_with_column = set(column_tables.get(normalized_column, []))
        if len(tables_with_column) < 2:
            continue
        if len(referenced_tables.intersection(tables_with_column)) < 2:
            continue
        if not _selects_unqualified_column(sql, normalized_column):
            continue
        issues.append(
            AnswerValidationIssue(
                code="ambiguous_same_name_column",
                message=(
                    f"The answer column `{column}` appears in multiple referenced tables "
                    f"{sorted(tables_with_column)}. Qualify the intended source column in the "
                    "query or join through the entity table before submitting the answer."
                ),
            )
        )

    return issues
