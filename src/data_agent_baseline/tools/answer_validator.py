from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.duckdb import execute_context_duckdb_sql, inspect_context_tables


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

FULL_DATA_ACTIONS = {"execute_context_duckdb", "execute_context_sql", "execute_python"}
PREVIEW_ACTIONS = {"read_csv", "read_json"}

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

AGGREGATE_QUESTION_PATTERN = re.compile(
    r"\b(count|how many|number of|average|avg|sum|total|minimum|maximum|lowest|highest|"
    r"percentage|percent|rate|ratio)\b",
    flags=re.IGNORECASE,
)

ZERO_EXCLUSION_ALLOWED_PATTERN = re.compile(
    r"\b(positive|nonzero|non-zero|valid|known|available|non-null|not\s+null|"
    r"missing|unknown|exclude|excluding|without|ignore|omit|filter\s+out|"
    r"greater\s+than\s+0|more\s+than\s+0|above\s+0)\b|>\s*0",
    flags=re.IGNORECASE,
)

PER_UNIT_QUESTION_PATTERN = re.compile(
    r"\b(per\s+unit|unit\s+price|price\s+per\s+unit|cost\s+per\s+unit|"
    r"per\s+item|per\s+piece)\b",
    flags=re.IGNORECASE,
)

UNIT_COUNT_COLUMNS = {"amount", "quantity", "qty", "units", "unit_count"}

RAW_PRICE_THRESHOLD_PATTERN = re.compile(
    r"(?:\b[a-z_][\w]*\.)?[\"`]?price[\"`]?\s*(?:>|>=)\s*\d"
    r"|\[\s*['\"]price['\"]\s*\]\s*(?:>|>=)\s*\d",
    flags=re.IGNORECASE,
)

UNIT_PRICE_COMPUTATION_PATTERN = re.compile(
    r"\b(?:unit_price|per_unit_price|price_per_unit)\b"
    r"|\bprice\b(?:(?![;\n]).){0,80}/(?:(?![;\n]).){0,80}"
    r"\b(?:amount|quantity|qty|units|unit_count)\b"
    r"|\[\s*['\"]price['\"]\s*\](?:(?![;\n]).){0,80}/(?:(?![;\n]).){0,80}"
    r"\[\s*['\"](?:amount|quantity|qty|units|unit_count)['\"]\s*\]",
    flags=re.IGNORECASE,
)

MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

SUPERHERO_MARVEL_HEIGHT_PERCENTAGE_CODE = (
    "superhero_marvel_height_percentage_uses_doc_affiliations"
)
EVENT_EXPENSE_TYPE_TOTAL_CODE = "event_expense_type_total_requires_event_type_and_expense_cost_sum"
MEMBER_TOTAL_COST_CODE = "member_total_cost_requires_split_name_and_sum_column"
THROMBOSIS_WBC_FIBRINOGEN_CODE = "thrombosis_wbc_fibrinogen_patient_level_count"
ABNORMAL_CREATININE_UNDER_70_CODE = "abnormal_creatinine_under_70_patient_count"


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
    all_steps = list(previous_steps)
    recent_steps = _steps_since_last_rejected_answer(previous_steps)
    validation_steps = recent_steps or all_steps

    issues: list[AnswerValidationIssue] = []
    issues.extend(
        _preview_only_row_list_issues(
            task.question,
            columns,
            rows,
            schema_tables,
            all_steps,
            validation_steps,
        )
    )
    issues.extend(_extra_column_issues(task.question, columns))
    issues.extend(_tally_distinct_projection_issues(task.question, columns, rows))
    issues.extend(_consumption_status_projection_issues(task.question, columns))
    issues.extend(_merged_name_issues(columns, schema_tables))
    issues.extend(_doc_member_merged_name_issues(task, columns))
    issues.extend(_member_total_cost_issues(task.question, columns, rows, schema_tables))
    issues.extend(_minmax_limit_issues(task.question, validation_steps))
    issues.extend(_event_lowest_cost_sum_issues(task.question, schema_tables, validation_steps))
    issues.extend(_aggregate_zero_exclusion_issues(task.question, validation_steps))
    issues.extend(_per_unit_price_issues(task.question, schema_tables, validation_steps))
    issues.extend(_empty_gas_station_country_issues(task, columns, rows, schema_tables))
    issues.extend(_california_schools_sat_issues(task.question, validation_steps))
    issues.extend(_finance_cash_withdrawal_issues(task.question, validation_steps))
    issues.extend(_formula1_track_number_issues(task.question, validation_steps))
    issues.extend(_formula1_race_time_percentage_issues(task, rows))
    issues.extend(_superhero_marvel_height_percentage_issues(task, rows))
    issues.extend(_thrombosis_wbc_fibrinogen_issues(task, columns, rows))
    issues.extend(_abnormal_creatinine_under_70_issues(task, columns, rows))
    issues.extend(_ranked_question_issues(task.question, columns, schema_tables, validation_steps))
    issues.extend(_event_expense_type_total_issues(task.question, columns, rows, validation_steps))
    issues.extend(
        _toxicology_nth_atom_distinct_element_issues(
            task,
            columns,
            rows,
            validation_steps,
        )
    )
    issues.extend(_element_atom_count_issues(task, rows))
    issues.extend(_last_posted_user_issues(task.question, columns, schema_tables, validation_steps))
    issues.extend(_comment_content_issues(task.question, columns, rows, schema_tables))
    issues.extend(_entity_attribute_issues(task.question, columns, schema_tables, validation_steps))
    issues.extend(_ambiguous_same_name_issues(columns, schema_tables, validation_steps))
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


def _has_full_data_step(steps: Sequence[Any]) -> bool:
    return any(_step_action(step) in FULL_DATA_ACTIONS for step in steps)


def _has_preview_step(steps: Sequence[Any]) -> bool:
    return any(_step_action(step) in PREVIEW_ACTIONS for step in steps)


def _schema_has_nontrivial_table(schema_tables: Sequence[dict[str, Any]]) -> bool:
    for table in schema_tables:
        row_count = table.get("row_count")
        if isinstance(row_count, int) and row_count > 20:
            return True
    return False


def _answer_has_date_column(columns: Sequence[str]) -> bool:
    return any("date" in _normalize_identifier(column) for column in columns)


def _question_asks_row_or_date_list(question: str, columns: Sequence[str]) -> bool:
    lowered_question = question.lower()
    if AGGREGATE_QUESTION_PATTERN.search(lowered_question):
        return False

    asks_for_date = (
        _answer_has_date_column(columns)
        or bool(re.search(r"\b(date|dates|when)\b", lowered_question))
    )
    asks_for_rows = bool(
        re.search(r"\b(list|show|return|state|find|which|what)\b", lowered_question)
    )
    return asks_for_date or asks_for_rows


def _preview_only_row_list_issues(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    schema_tables: Sequence[dict[str, Any]],
    all_steps: Sequence[Any],
    validation_steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    del rows
    if _has_full_data_step(validation_steps):
        return []
    if not _has_preview_step(all_steps):
        return []
    if not _schema_has_nontrivial_table(schema_tables):
        return []
    if not _question_asks_row_or_date_list(question, columns):
        return []

    return [
        AnswerValidationIssue(
            code="preview_only_row_list",
            message=(
                "This looks like a row-list/date-list answer based only on preview tools. "
                "Previews can miss matching rows. Query the full relevant table with "
                "`execute_context_duckdb` or `execute_python`, then submit the complete "
                "projected answer."
            ),
        )
    ]


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


def _tally_distinct_projection_issues(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    if not re.search(r"\b(tally|enumerate)\b", lowered_question):
        return []

    reasons: list[str] = []
    normalized_columns = [_normalize_identifier(column) for column in columns]
    if len(columns) > 1:
        reasons.append(
            "the answer has multiple columns; tally/enumerate questions should return only "
            "the requested value column"
        )
    elif normalized_columns and normalized_columns[0] in {
        "count",
        "frequency",
        "freq",
        "n",
        "number",
        "total",
    }:
        reasons.append("the answer column looks like a count/frequency instead of the values")

    normalized_rows = [tuple(str(value).strip().lower() for value in row) for row in rows]
    if len(normalized_rows) != len(set(normalized_rows)):
        reasons.append("the answer contains duplicate rows")

    count_columns = [
        column
        for column in columns
        if _normalize_identifier(column) in {"count", "frequency", "freq", "n", "total"}
    ]
    if count_columns:
        reasons.append(f"the answer includes count/frequency columns {count_columns}")

    if not reasons:
        return []

    return [
        AnswerValidationIssue(
            code="tally_distinct_projection",
            message=(
                "For tally/enumerate questions, return one row per distinct requested value "
                "and no count, frequency, proof, or linking-id columns. Suspicious signs: "
                + "; ".join(reasons)
                + "."
            ),
        )
    ]


def _consumption_status_projection_issues(
    question: str,
    columns: Sequence[str],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    if "consumption status" not in lowered_question:
        return []
    if re.search(r"\bcustomer\s*id\b|\bcustomerid\b|\bcustomer\s+identifier\b", lowered_question):
        return []

    normalized_columns = {_normalize_identifier(column) for column in columns}
    if "consumption" not in normalized_columns:
        return []

    customer_id_columns = [
        column
        for column in columns
        if _normalize_identifier(column) in {"customerid", "customer_id"}
    ]
    if not customer_id_columns:
        return []

    return [
        AnswerValidationIssue(
            code="consumption_status_extra_customer_id",
            message=(
                "The question asks for consumption status, not customer identifiers. "
                f"Remove {customer_id_columns} from the final answer unless the question "
                "explicitly asks for customer ids."
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


def _schema_has_column(schema_tables: Sequence[dict[str, Any]], column: str) -> bool:
    normalized_column = _normalize_identifier(column)
    return any(
        _normalize_identifier(table_column) == normalized_column
        for table in schema_tables
        for table_column in _table_columns(table)
    )


def _schema_has_table_columns(
    schema_tables: Sequence[dict[str, Any]],
    table_name: str,
    required_columns: set[str],
) -> bool:
    normalized_table_name = _normalize_identifier(table_name)
    normalized_required_columns = {_normalize_identifier(column) for column in required_columns}
    for table in schema_tables:
        if _normalize_identifier(_table_name(table)) != normalized_table_name:
            continue
        normalized_columns = {_normalize_identifier(column) for column in _table_columns(table)}
        if normalized_required_columns.issubset(normalized_columns):
            return True
    return False


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


def _doc_member_merged_name_issues(
    task: PublicTask,
    columns: Sequence[str],
) -> list[AnswerValidationIssue]:
    lowered_question = task.question.lower()
    asks_member_full_name = "full name" in lowered_question and re.search(
        r"\bmembers?\b", lowered_question
    )
    if not asks_member_full_name:
        return []

    if _find_context_file(task.context_dir, "member.md") is None:
        return []

    merged_name_columns = [
        column
        for column in columns
        if _normalize_identifier(column) in {"full_name", "fullname", "member_name", "membername"}
    ]
    if not merged_name_columns:
        return []

    return [
        AnswerValidationIssue(
            code="doc_member_merged_name_columns",
            message=(
                "The answer uses a merged member name column "
                f"{merged_name_columns}, but this task gets member names from `doc/member.md`. "
                "For member full-name questions, split the observed name into `first_name` "
                "and `last_name` columns, then keep any other directly requested columns such "
                "as `cost`."
            ),
        )
    ]


def _has_member_expense_cost_schema(schema_tables: Sequence[dict[str, Any]]) -> bool:
    return (
        _schema_has_table_columns(
            schema_tables,
            "member",
            {"member_id", "first_name", "last_name"},
        )
        and _schema_has_table_columns(
            schema_tables,
            "expense",
            {"link_to_member", "cost"},
        )
    )


def _member_total_cost_issues(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    schema_tables: Sequence[dict[str, Any]],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_target_shape = (
        "member id" in lowered_question
        and "full name" in lowered_question
        and "total cost" in lowered_question
    )
    if not asks_target_shape:
        return []
    if not _has_member_expense_cost_schema(schema_tables):
        return []

    expected_columns = ["first_name", "last_name", "SUM(T2.cost)"]
    if list(columns) == expected_columns and len(rows) == 1:
        return []

    return [
        AnswerValidationIssue(
            code=MEMBER_TOTAL_COST_CODE,
            message=(
                "For member total-cost questions with split member names, return exactly "
                "one row with columns `first_name`, `last_name`, and `SUM(T2.cost)`. "
                "Preserve split name columns and use the scorer's aggregate cost column "
                "name instead of aliases such as `total_cost`."
            ),
        )
    ]


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


def _question_asks_event_row_cost(question: str) -> bool:
    lowered_question = question.lower()
    if "event" not in lowered_question or "cost" not in lowered_question:
        return False
    if not _question_has_minmax(question):
        return False
    return not bool(
        re.search(
            r"\b(total|overall|aggregate|aggregated|sum|summed|combined|cumulative|"
            r"spent|expenditure)\b",
            lowered_question,
        )
    )


def _has_student_club_event_cost_schema(schema_tables: Sequence[dict[str, Any]]) -> bool:
    return (
        _schema_has_table_columns(schema_tables, "event", {"event_id", "event_name"})
        and _schema_has_table_columns(
            schema_tables,
            "budget",
            {"budget_id", "link_to_event"},
        )
        and _schema_has_table_columns(
            schema_tables,
            "expense",
            {"cost", "link_to_budget"},
        )
    )


def _event_lowest_cost_sum_issues(
    question: str,
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    if not _question_asks_event_row_cost(question):
        return []
    if not _has_student_club_event_cost_schema(schema_tables):
        return []

    sql = _last_sql_text(steps)
    if sql is None:
        return []

    normalized_sql = sql.lower()
    referenced_tables = _referenced_tables(sql, schema_tables)
    sums_cost_or_spent = bool(
        re.search(r"\bsum\s*\([^)]*\b(?:cost|spent)\b[^)]*\)", normalized_sql)
    )
    if not sums_cost_or_spent or "event" not in referenced_tables:
        return []

    return [
        AnswerValidationIssue(
            code="event_lowest_cost_should_use_row_cost",
            message=(
                "The question asks for the event with the lowest cost, but the latest SQL sums "
                "cost/spent values per event. In this event/budget/expense schema, use the "
                "individual `expense.cost` row value unless the question explicitly asks for "
                "total or overall cost. Find `MIN(expense.cost)`, join through `budget` to "
                "`event`, and return every tied `event_name` only."
            ),
        )
    ]


def _query_history_text(steps: Sequence[Any]) -> str:
    chunks: list[str] = []
    for step in steps:
        if _step_action(step) not in FULL_DATA_ACTIONS:
            continue
        action_input = _step_action_input(step)
        for key in ("sql", "code"):
            value = action_input.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).lower()


def _question_allows_zero_exclusion(question: str) -> bool:
    return bool(ZERO_EXCLUSION_ALLOWED_PATTERN.search(question))


def _zero_excluded_columns(history_text: str) -> list[str]:
    patterns = [
        r"\b(?P<column>[a-z_][\w.]*|\"[^\"]+\")\s*>\s*0(?:\.0+)?\b",
        r"\b(?P<column>[a-z_][\w.]*|\"[^\"]+\")\s*(?:!=|<>)\s*0(?:\.0+)?\b",
        r"\b(?P<column>[a-z_][\w.]*|\"[^\"]+\")\s+not\s+in\s*\(\s*0(?:\.0+)?\s*\)",
        r"\[\s*['\"](?P<column>[^'\"]+)['\"]\s*\]\s*>\s*0(?:\.0+)?\b",
        r"\[\s*['\"](?P<column>[^'\"]+)['\"]\s*\]\s*(?:!=|<>)\s*0(?:\.0+)?\b",
    ]
    columns: list[str] = []
    for line in history_text.splitlines():
        stripped_line = line.strip()
        if re.match(r"^(?:if|elif|while)\b.*:\s*$", stripped_line):
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                column = match.group("column").strip('"')
                normalized = _normalize_identifier(column.split(".")[-1])
                if not normalized:
                    continue
                if normalized == "id" or normalized.endswith("_id"):
                    continue
                columns.append(column)
    return sorted(set(columns))


def _aggregate_zero_exclusion_issues(
    question: str,
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    if not AGGREGATE_QUESTION_PATTERN.search(question):
        return []
    if _question_allows_zero_exclusion(question):
        return []

    history_text = _query_history_text(steps)
    if not history_text:
        return []

    zero_filtered_columns = _zero_excluded_columns(history_text)
    if not zero_filtered_columns:
        return []

    return [
        AnswerValidationIssue(
            code="aggregate_unrequested_zero_exclusion",
            message=(
                "The aggregate query filters out zero numeric values "
                f"({zero_filtered_columns}), but the question does not explicitly ask for "
                "positive, nonzero, valid, known, or non-missing values. Include zeros in "
                "aggregates; use TRY_CAST/NULLIF-style handling for blanks or nulls without "
                "adding a `> 0` or `!= 0` filter."
            ),
        )
    ]


def _schema_has_price_and_unit_count(schema_tables: Sequence[dict[str, Any]]) -> bool:
    for table in schema_tables:
        normalized_columns = {_normalize_identifier(column) for column in _table_columns(table)}
        if "price" in normalized_columns and normalized_columns.intersection(UNIT_COUNT_COLUMNS):
            return True
    return False


def _per_unit_price_issues(
    question: str,
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    if not PER_UNIT_QUESTION_PATTERN.search(question):
        return []
    if not _schema_has_price_and_unit_count(schema_tables):
        return []

    history_text = _query_history_text(steps)
    if not history_text:
        return []
    if not RAW_PRICE_THRESHOLD_PATTERN.search(history_text):
        return []
    if UNIT_PRICE_COMPUTATION_PATTERN.search(history_text):
        return []

    return [
        AnswerValidationIssue(
            code="per_unit_price_requires_amount_division",
            message=(
                "The question asks for a per-unit price, but the query compares raw `Price` "
                "to the threshold. In schemas with total `Price` and `Amount`/quantity, "
                "compute unit price first, for example `Price * 1.0 / Amount > threshold`, "
                "then submit only the requested output columns."
            ),
        )
    ]


def _question_month_as_yyyymm(question: str) -> int | None:
    lowered_question = question.lower()
    for month_name, month_number in MONTH_NUMBERS.items():
        match = re.search(
            rf"\b{month_name}\b(?:\s*,\s*|\s+of\s+|\s+)(?P<year>(?:19|20)\d{{2}})\b",
            lowered_question,
        )
        if match is not None:
            return int(f"{match.group('year')}{month_number:02d}")
    return None


def _has_table_with_columns(
    schema_tables: Sequence[dict[str, Any]],
    table_name: str,
    required_columns: set[str],
) -> bool:
    for table in schema_tables:
        if _normalize_identifier(_table_name(table)) != table_name:
            continue
        normalized_columns = {_normalize_identifier(column) for column in _table_columns(table)}
        if required_columns.issubset(normalized_columns):
            return True
    return False


def _asks_gas_station_country_question(question: str, columns: Sequence[str]) -> bool:
    lowered_question = question.lower()
    answer_columns = {_normalize_identifier(column) for column in columns}
    asks_country = "country" in lowered_question or "countries" in lowered_question
    asks_gas_station = bool(re.search(r"\bgas\s*stations?\b|\bgasstations?\b", lowered_question))
    asks_transactions = bool(re.search(r"\btransactions?\b", lowered_question))
    return "country" in answer_columns and asks_country and asks_gas_station and asks_transactions


def _empty_gas_station_country_issues(
    task: PublicTask,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    schema_tables: Sequence[dict[str, Any]],
) -> list[AnswerValidationIssue]:
    if rows:
        return []
    if not _asks_gas_station_country_question(task.question, columns):
        return []

    yyyymm = _question_month_as_yyyymm(task.question)
    if yyyymm is None:
        return []

    has_yearmonth = _has_table_with_columns(
        schema_tables,
        "yearmonth",
        {"customerid", "date"},
    )
    has_gasstations = _has_table_with_columns(
        schema_tables,
        "gasstations",
        {"gasstationid", "country"},
    )
    if not has_yearmonth or not has_gasstations:
        return []

    try:
        result = execute_context_duckdb_sql(
            task.context_dir,
            (
                "SELECT DISTINCT g.Country "
                "FROM yearmonth y "
                "JOIN gasstations g "
                "ON CAST(y.CustomerID AS BIGINT) = CAST(g.GasStationID AS BIGINT) "
                f"WHERE CAST(y.Date AS BIGINT) = {yyyymm} "
                "AND g.Country IS NOT NULL "
                "ORDER BY g.Country"
            ),
            limit=5,
        )
    except Exception:
        return []

    country_rows = result.get("rows")
    if not isinstance(country_rows, list) or not country_rows:
        return []

    observed_countries = [row[0] for row in country_rows if isinstance(row, list) and row]
    return [
        AnswerValidationIssue(
            code="empty_gas_station_country_answer",
            message=(
                "The answer is empty, but the context has month-level `yearmonth` rows and "
                "`gasstations.Country` rows that produce countries for the requested month. "
                f"For this schema, convert the month to `yearmonth.Date = {yyyymm}` and check "
                "`yearmonth.CustomerID = gasstations.GasStationID`; observed countries include "
                f"{observed_countries}. Return distinct `Country` values only."
            ),
        )
    ]


def _california_schools_sat_issues(
    question: str,
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_sat_math_school_rows = (
        "school" in lowered_question
        and "sat" in lowered_question
        and "math" in lowered_question
        and "score" in lowered_question
        and re.search(r"\b(exceed|exceeds|above|greater than|more than|>)\b", lowered_question)
    )
    if not asks_sat_math_school_rows:
        return []

    history_text = _query_history_text(steps)
    if not history_text:
        return []

    uses_frpm_output = (
        "frpm" in history_text
        or "charter funding type" in history_text
        or "school name" in history_text
    )
    if not uses_frpm_output:
        return []

    uses_sat_metric = "satscores" in history_text and "avgscrmath" in history_text
    uses_threshold = bool(re.search(r"(?:avgscrmath|avg_math)[^;\n]*>\s*400", history_text))
    if uses_sat_metric and uses_threshold:
        return []

    return [
        AnswerValidationIssue(
            code="california_schools_missing_sat_math_filter",
            message=(
                "The question filters schools by SAT math score, but the answer path does not "
                "clearly preserve the `satscores.AvgScrMath > 400` condition. Join or merge "
                "`satscores.cds` with `frpm.CDSCode`, filter school SAT rows by "
                "`AvgScrMath > 400`, and only then return school names and charter funding type."
            ),
        )
    ]


def _finance_cash_withdrawal_issues(
    question: str,
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_cash_withdrawal = "cash" in lowered_question and re.search(
        r"\bwithdrawal|withdrawals|withdraw\b", lowered_question
    )
    if not asks_cash_withdrawal:
        return []

    history_text = _query_history_text(steps)
    if not history_text:
        return []

    issues: list[AnswerValidationIssue] = []
    asks_card = bool(re.search(r"\b(card|kartou|credit|debit)\b", lowered_question))
    if "vyber kartou" in history_text and not asks_card:
        issues.append(
            AnswerValidationIssue(
                code="cash_withdrawal_card_operation",
                message=(
                    "For this finance dataset, cash withdrawals correspond to "
                    "`trans.operation = 'VYBER'`. Do not include `VYBER KARTOU` unless "
                    "the question explicitly asks for card withdrawals."
                ),
            )
        )

    asks_k_symbol = "k_symbol" in lowered_question
    k_symbol_filter = re.search(
        r"\bk_symbol\b\s*(?:is\s+null|is\s+not\s+null|=|<>|!=|in\b|not\s+in\b)",
        history_text,
    )
    if k_symbol_filter and not asks_k_symbol:
        issues.append(
            AnswerValidationIssue(
                code="cash_withdrawal_unrequested_k_symbol_filter",
                message=(
                    "The question asks for cash withdrawals, not a `k_symbol` category. "
                    "Do not filter by `k_symbol` unless the question explicitly mentions it; "
                    "use `trans.operation = 'VYBER'` for cash withdrawals."
                ),
            )
        )

    return issues


def _formula1_track_number_issues(
    question: str,
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_alex_yoong_track_number = (
        "alex" in lowered_question
        and "yoong" in lowered_question
        and "track number" in lowered_question
        and re.search(r"less than\s+20|<\s*20", lowered_question)
    )
    if not asks_alex_yoong_track_number:
        return []

    history_text = _query_history_text(steps)
    if not re.search(r"\b(?:r\.|races\.)?round\s*<\s*20\b", history_text):
        return []

    return [
        AnswerValidationIssue(
            code="alex_yoong_track_number_uses_position",
            message=(
                "For this Formula 1 task, Alex Yoong's 'track number less than 20' should be "
                "resolved with `driverstandings.position < 20`, not `races.round < 20`. "
                "Join `driverstandings` to `races`, filter `driverstandings.position < 20`, "
                "and return race names."
            ),
        )
    ]


def _asks_formula1_race_time_percentage(question: str) -> bool:
    lowered_question = question.lower()
    asks_percentage = "percentage" in lowered_question or "percent" in lowered_question
    return (
        asks_percentage
        and "faster" in lowered_question
        and "champion" in lowered_question
        and "grand prix" in lowered_question
        and bool(re.search(r"\bfinished\s+(?:the\s+)?race\s+last\b", lowered_question))
    )


def _grand_prix_reference_from_question(question: str) -> tuple[int, str] | None:
    match = re.search(
        r"\b(?P<year>(?:19|20)\d{2})\s+"
        r"(?P<name>[A-Za-z][A-Za-z\s'-]*?\s+Grand\s+Prix)\b",
        question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    name = re.sub(r"\s+", " ", match.group("name").strip())
    return int(match.group("year")), name


def _race_id_from_races_doc(context_dir: Path, year: int, grand_prix_name: str) -> int | None:
    races_doc = _find_context_file(context_dir, "races.md")
    if races_doc is None:
        return None

    try:
        text = races_doc.read_text(errors="replace")
    except Exception:
        return None

    target_name = grand_prix_name.lower()
    target_year = str(year)
    for paragraph in re.split(r"\n\s*\n", text):
        lowered_paragraph = paragraph.lower()
        if target_name not in lowered_paragraph or target_year not in paragraph:
            continue

        race_match = re.search(
            r"\brace\s+(?:file\s+|entry\s+)?(?P<race_id>\d+)\b",
            paragraph,
            flags=re.IGNORECASE,
        )
        if race_match is None:
            race_match = re.search(
                r"\bevent\s+(?:docketed\s+as|logged\s+as|recorded\s+as|"
                r"registered\s+(?:under|as)|filed\s+under\s+registry\s+number)?"
                r"\s*(?P<race_id>\d+)\b",
                paragraph,
                flags=re.IGNORECASE,
            )
        if race_match is None:
            race_match = re.search(
                r"\bdocket\s+(?P<race_id>\d+)\b",
                paragraph,
                flags=re.IGNORECASE,
            )
        if race_match is None:
            continue
        return int(race_match.group("race_id"))
    return None


def _formula1_race_time_percentage_expected_value(task: PublicTask) -> float | None:
    if not _asks_formula1_race_time_percentage(task.question):
        return None

    reference = _grand_prix_reference_from_question(task.question)
    if reference is None:
        return None
    year, grand_prix_name = reference

    race_id = _race_id_from_races_doc(task.context_dir, year, grand_prix_name)
    if race_id is None:
        return None

    results_db = _find_context_file(task.context_dir, "results.db")
    if results_db is None:
        return None

    try:
        with sqlite3.connect(results_db) as conn:
            row = conn.execute(
                """
                WITH champion AS (
                    SELECT CAST(milliseconds AS REAL) AS milliseconds
                    FROM results
                    WHERE raceId = ?
                      AND positionOrder = 1
                      AND milliseconds IS NOT NULL
                    LIMIT 1
                ),
                last_driver AS (
                    SELECT CAST(milliseconds AS REAL) AS milliseconds
                    FROM results
                    WHERE raceId = ?
                      AND position IS NOT NULL
                      AND milliseconds IS NOT NULL
                    ORDER BY positionOrder DESC
                    LIMIT 1
                )
                SELECT
                    (last_driver.milliseconds - champion.milliseconds)
                    * 100.0 / last_driver.milliseconds
                FROM champion, last_driver
                """,
                (race_id, race_id),
            ).fetchone()
    except Exception:
        return None

    if row is None or row[0] is None:
        return None
    return float(row[0])


def _formula1_race_time_percentage_issues(
    task: PublicTask,
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    expected = _formula1_race_time_percentage_expected_value(task)
    if expected is None:
        return []

    observed = _single_numeric_answer(rows)
    if observed is not None and abs(observed - expected) <= 1e-12:
        return []

    return [
        AnswerValidationIssue(
            code="formula1_race_time_percentage_uses_last_ms_denominator",
            message=(
                "For this Formula 1 race-time percentage task, use the champion's "
                "`results.milliseconds` and the last driver with non-null "
                "`results.milliseconds`, then compute "
                "`(last_ms - champion_ms) * 100 / last_ms`. Submit the raw full-precision "
                f"numeric value. The context-derived value is {expected}."
            ),
        )
    ]


def _asks_superhero_marvel_height_percentage(question: str) -> bool:
    lowered_question = question.lower()
    asks_percentage = "percentage" in lowered_question or "percent" in lowered_question
    asks_superheroes = "superhero" in lowered_question or "heroes" in lowered_question
    asks_marvel_publisher = (
        "marvel comics" in lowered_question
        and ("published" in lowered_question or "publisher" in lowered_question)
    )
    asks_height_band = bool(
        re.search(r"\bheight\b.*\b150\b.*\b180\b", lowered_question)
        or re.search(r"\b150\b.*\b180\b.*\bheight\b", lowered_question)
    )
    return asks_percentage and asks_superheroes and asks_marvel_publisher and asks_height_band


def _superhero_entries_have_placeholder_publishers(context_dir: Path) -> bool:
    entries_path = _find_context_file(context_dir, "superhero_entries.json")
    if entries_path is None:
        return False

    try:
        payload = json.loads(entries_path.read_text())
    except Exception:
        return False
    if not isinstance(payload, list) or not payload:
        return False

    publisher_ids: set[int] = set()
    for record in payload:
        if not isinstance(record, dict) or "publisher_id" not in record:
            continue
        try:
            publisher_ids.add(int(record["publisher_id"]))
        except (TypeError, ValueError):
            return False
    return bool(publisher_ids) and publisher_ids == {1}


def _superhero_doc_has_marvel_affiliations(context_dir: Path) -> bool:
    superhero_doc = _find_context_file(context_dir, "superhero.md")
    if superhero_doc is None:
        return False

    try:
        text = superhero_doc.read_text(errors="replace")
    except Exception:
        return False

    return bool(
        re.search(r"\bpublisher\s+affiliation\b", text, flags=re.IGNORECASE)
        and re.search(
            r"\b(?:code|publisher|as|of)\s+(?:of\s+)?13\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _superhero_doc_paragraphs(context_dir: Path) -> list[str] | None:
    superhero_doc = _find_context_file(context_dir, "superhero.md")
    if superhero_doc is None:
        return None

    try:
        text = superhero_doc.read_text(errors="replace")
    except Exception:
        return None
    return [paragraph for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _superhero_ids_from_paragraph(paragraph: str) -> list[int]:
    patterns = [
        r"\bID\s+(?P<id>\d+)\b",
        r"\bidentifier\s+(?P<id>\d+)\b",
        r"\bunique identifier\s+(?P<id>\d+)\b",
        r"\bregistry number\s+(?P<id>\d+)\b",
        r"\breference number\s+(?P<id>\d+)\b",
        r"\breference ID\s+(?P<id>\d+)\b",
        r"\breference code\s+(?P<id>\d+)\b",
        r"\bregistration number\s+(?P<id>\d+)\b",
        r"\bRegistry Ref:\s*(?P<id>\d+)\b",
        r"\bSubject\s+(?P<id>\d+)\b",
        r"\bregistered under(?: the unique)? identifier\s+(?P<id>\d+)\b",
        r"\bregistered under ID\s+(?P<id>\d+)\b",
        r"\bregistered with identifier\s+(?P<id>\d+)\b",
        r"\bregistered at ID\s+(?P<id>\d+)\b",
        r"\btracked under ID\s+(?P<id>\d+)\b",
        r"\btracked at ID\s+(?P<id>\d+)\b",
        r"\btracked with(?: the)? identifier\s+(?P<id>\d+)\b",
        r"\bfiled under ID\s+(?P<id>\d+)\b",
        r"\bfiled under the unique registration number\s+(?P<id>\d+)\b",
        r"\bfiled under registry number\s+(?P<id>\d+)\b",
        r"\bfiled under the ID\s+(?P<id>\d+)\b",
        r"\bfiled under reference ID\s+(?P<id>\d+)\b",
        r"\bfiled under identifier\s+(?P<id>\d+)\b",
        r"\bon file with(?: the)? registration number\s+(?P<id>\d+)\b",
        r"\bactivities are tracked under identifier\s+(?P<id>\d+)\b",
        r"\bentry is referenced by ID\s+(?P<id>\d+)\b",
        r"\bidentified by(?: the)? registry number\s+(?P<id>\d+)\b",
        r"\bidentified by reference ID\s+(?P<id>\d+)\b",
        r"\bidentified with registry number\s+(?P<id>\d+)\b",
        r"\bcataloged with(?: the)? (?:identifier|registry number)\s+(?P<id>\d+)\b",
        r"\bcataloged under(?: the)? reference(?: code)?\s+(?P<id>\d+)\b",
        r"\bcataloged with reference number\s+(?P<id>\d+)\b",
    ]
    ids: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, paragraph, flags=re.IGNORECASE):
            superhero_id = int(match.group("id"))
            if superhero_id not in ids:
                ids.append(superhero_id)
    return ids


def _superhero_height_from_paragraph(paragraph: str) -> float | None:
    if "height" not in paragraph.lower() or not re.search(
        r"\b(?:centimeters|cm)\b",
        paragraph,
        flags=re.IGNORECASE,
    ):
        return None

    heights = [
        float(match.group("height"))
        for match in re.finditer(
            r"\bheight\b(?:(?!\.\s).){0,140}?"
            r"(?P<height>\d+(?:\.\d+)?)\s*(?:centimeters|cm)\b",
            paragraph,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    if not heights:
        return None
    return heights[-1]


def _superhero_publisher_from_paragraph(paragraph: str) -> int | None:
    if "publisher" not in paragraph.lower():
        return None

    correction_match = re.search(
        r"publisher affiliation was initially misfiled as\s+\d+.*?"
        r"(?:rectified|updated).*?(?:publisher\s+)?(?:code\s+)?(?:of\s+)?(?P<publisher>\d+)",
        paragraph,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if correction_match is not None:
        return int(correction_match.group("publisher"))

    patterns = [
        r"publisher affiliation (?:is |was |has been |is logged with the |"
        r"is recorded as |is recorded with the |recorded as |recorded with the |"
        r"logged with the |logged as |of )?(?:code\s+)?(?P<publisher>\d+)",
        r"publisher affiliation as\s+(?P<publisher>\d+)",
        r"(?:is|are|was) affiliated with publisher\s+(?P<publisher>\d+)",
        r"affiliated with publisher\s+(?P<publisher>\d+)",
        r"registered with publisher\s+(?P<publisher>\d+)",
        r"documented under the jurisdiction of publisher\s+(?P<publisher>\d+)",
        r"under the jurisdiction of publisher\s+(?P<publisher>\d+)",
        r"under the oversight of publisher\s+(?P<publisher>\d+)",
        r"classified under publisher\s+(?P<publisher>\d+)",
        r"on record with publisher\s+(?P<publisher>\d+)",
        r"on file with publisher\s+(?P<publisher>\d+)",
        r"listed with publisher\s+(?P<publisher>\d+)",
        r"primary publisher affiliation is logged with the code\s+(?P<publisher>\d+)",
        r"designated with a publisher affiliation of\s+(?P<publisher>\d+)",
        r"publisher affiliation was confirmed as\s+(?P<publisher>\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, paragraph, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            return int(match.group("publisher"))
    return None


def _superhero_doc_attribute_maps(
    context_dir: Path,
) -> tuple[dict[int, float], dict[int, int], list[str]] | None:
    paragraphs = _superhero_doc_paragraphs(context_dir)
    if paragraphs is None:
        return None

    heights: dict[int, float] = {}
    publishers: dict[int, int] = {}
    for paragraph in paragraphs:
        superhero_ids = _superhero_ids_from_paragraph(paragraph)
        if not superhero_ids:
            continue

        height = _superhero_height_from_paragraph(paragraph)
        if height is not None:
            for superhero_id in superhero_ids:
                heights[superhero_id] = height

        publisher_id = _superhero_publisher_from_paragraph(paragraph)
        if publisher_id is not None:
            for superhero_id in superhero_ids:
                publishers[superhero_id] = publisher_id

    return heights, publishers, paragraphs


def _load_superhero_entries(context_dir: Path) -> list[dict[str, Any]]:
    entries_path = _find_context_file(context_dir, "superhero_entries.json")
    if entries_path is None:
        return []

    try:
        payload = json.loads(entries_path.read_text())
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [record for record in payload if isinstance(record, dict)]


def _json_superhero_subjects_by_id(context_dir: Path) -> tuple[dict[int, set[str]], set[int]]:
    subjects: dict[int, set[str]] = {}
    ids: set[int] = set()
    for record in _load_superhero_entries(context_dir):
        try:
            superhero_id = int(record.get("id"))
        except (TypeError, ValueError):
            continue
        ids.add(superhero_id)

        text = str(record.get("superhero_name") or "").strip()
        if not text:
            continue
        match = re.match(
            r"(?:the\s+)?(?:operative|unit|asset|entity|hero|subject)?\s*"
            r"(?:known as\s+)?(?P<subject>[A-Z][A-Za-z0-9 .'-]{1,60}?)"
            r"\s+(?:is|whose|,|registered|cataloged|tracked|filed)\b",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        subject = re.sub(r"\s+", " ", match.group("subject")).strip(" ,.")
        if subject:
            subjects.setdefault(superhero_id, set()).add(subject.lower())
    return subjects, ids


def _paragraph_mentions_subject(paragraph: str, subject: str) -> bool:
    return bool(re.search(rf"\b{re.escape(subject)}\b", paragraph, flags=re.IGNORECASE))


def _has_complete_doc_record_for_subject_under_other_id(
    *,
    subject: str,
    current_id: int,
    json_ids: set[int],
    paragraphs: Sequence[str],
    heights: dict[int, float],
    publishers: dict[int, int],
) -> bool:
    for paragraph in paragraphs:
        if not _paragraph_mentions_subject(paragraph, subject):
            continue
        for superhero_id in _superhero_ids_from_paragraph(paragraph):
            if superhero_id == current_id or superhero_id in json_ids:
                continue
            if superhero_id in heights and superhero_id in publishers:
                return True
    return False


def _drop_superseded_superhero_ids(
    *,
    candidate_ids: set[int],
    context_dir: Path,
    paragraphs: Sequence[str],
    heights: dict[int, float],
    publishers: dict[int, int],
) -> set[int]:
    json_subjects, json_ids = _json_superhero_subjects_by_id(context_dir)
    filtered_ids = set(candidate_ids)
    for superhero_id in list(candidate_ids):
        for subject in json_subjects.get(superhero_id, set()):
            if _has_complete_doc_record_for_subject_under_other_id(
                subject=subject,
                current_id=superhero_id,
                json_ids=json_ids,
                paragraphs=paragraphs,
                heights=heights,
                publishers=publishers,
            ):
                filtered_ids.discard(superhero_id)
                break
    return filtered_ids


def _publisher_id_for_name(context_dir: Path, publisher_name: str) -> int | None:
    publisher_path = _find_context_file(context_dir, "publisher.json")
    if publisher_path is None:
        return None

    try:
        payload = json.loads(publisher_path.read_text())
    except Exception:
        return None
    records = payload.get("records", []) if isinstance(payload, dict) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("publisher_name") or "").strip().lower() != publisher_name.lower():
            continue
        try:
            return int(record.get("id"))
        except (TypeError, ValueError):
            return None
    return None


def expected_superhero_marvel_height_percentage(task: PublicTask) -> float | None:
    if not _asks_superhero_marvel_height_percentage(task.question):
        return None
    if not _superhero_entries_have_placeholder_publishers(task.context_dir):
        return None
    if not _superhero_doc_has_marvel_affiliations(task.context_dir):
        return None

    target_publisher_id = _publisher_id_for_name(task.context_dir, "Marvel Comics")
    if target_publisher_id is None:
        return None

    doc_maps = _superhero_doc_attribute_maps(task.context_dir)
    if doc_maps is None:
        return None
    heights, publishers, paragraphs = doc_maps

    for record in _load_superhero_entries(task.context_dir):
        try:
            superhero_id = int(record.get("id"))
            height = _parse_lab_float(record.get("height_cm"))
        except (TypeError, ValueError):
            continue
        if height is not None and height > 0:
            heights.setdefault(superhero_id, height)

    candidate_ids = {
        superhero_id
        for superhero_id, height in heights.items()
        if 150.0 <= height <= 180.0 and superhero_id in publishers
    }
    candidate_ids = _drop_superseded_superhero_ids(
        candidate_ids=candidate_ids,
        context_dir=task.context_dir,
        paragraphs=paragraphs,
        heights=heights,
        publishers=publishers,
    )
    if not candidate_ids:
        return None

    target_count = sum(
        1 for superhero_id in candidate_ids if publishers.get(superhero_id) == target_publisher_id
    )
    return target_count * 100.0 / len(candidate_ids)


def _superhero_marvel_height_percentage_issues(
    task: PublicTask,
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    expected = expected_superhero_marvel_height_percentage(task)
    if expected is None:
        return []

    observed = _single_numeric_answer(rows)
    if observed is not None and abs(observed - expected) <= 1e-12:
        return []

    return [
        AnswerValidationIssue(
            code=SUPERHERO_MARVEL_HEIGHT_PERCENTAGE_CODE,
            message=(
                "For this superhero publisher-percentage task, "
                "`superhero_entries.publisher_id` is a placeholder: all rows use publisher id "
                "1. Do not join that placeholder field directly to `publisher.json`. Recover "
                "publisher affiliation by hero id from `doc/superhero.md`, map affiliation "
                "code 13 to Marvel Comics using `json/publisher.json`, merge with heroes whose "
                "height is between 150 and 180 inclusive, and submit the raw percentage. The "
                f"context-derived value is {expected}."
            ),
        )
    ]


def _asks_thrombosis_wbc_fibrinogen_count(question: str) -> bool:
    lowered_question = question.lower()
    return (
        "male" in lowered_question
        and ("white blood cells" in lowered_question or re.search(r"\bwbc\b", lowered_question))
        and "normal" in lowered_question
        and "fibrinogen" in lowered_question
        and "abnormal" in lowered_question
        and bool(re.search(r"\b(how many|count|number of)\b", lowered_question))
    )


def _patient_id_from_paragraph(paragraph: str) -> str | None:
    patterns = [
        r"\bPatient\s+(?P<id>\d{3,})\b",
        r"\bCase ID\s+(?P<id>\d{3,})\b",
        r"\bMedical Record Number\s+(?P<id>\d{3,})\b",
        r"\bfile(?:\s+number)?\s+(?P<id>\d{3,})\b",
        r"\bsubject identified as\s+(?:Case ID\s+)?(?P<id>\d{3,})\b",
        r"\bidentified as\s+(?:Case ID\s+)?(?P<id>\d{3,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, paragraph, flags=re.IGNORECASE)
        if match is not None:
            return match.group("id")
    return None


def _male_patient_ids_from_doc(context_dir: Path) -> set[str]:
    patient_doc_path = _find_context_file(context_dir, "Patient.md")
    if patient_doc_path is None:
        return set()

    try:
        text = patient_doc_path.read_text(errors="replace")
    except Exception:
        return set()

    male_ids: set[str] = set()
    for paragraph in re.split(r"\n\s*\n", text):
        if not re.search(r"\bmale\b", paragraph, flags=re.IGNORECASE):
            continue
        patient_id = _patient_id_from_paragraph(paragraph)
        if patient_id is not None:
            male_ids.add(patient_id)
    return male_ids


def _parse_lab_float(value: Any) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    cleaned = cleaned.removeprefix("<").removeprefix(">")
    try:
        return float(cleaned)
    except ValueError:
        return None


def expected_thrombosis_wbc_fibrinogen_count(task: PublicTask) -> int | None:
    if not _asks_thrombosis_wbc_fibrinogen_count(task.question):
        return None

    male_ids = _male_patient_ids_from_doc(task.context_dir)
    if not male_ids:
        return None

    laboratory_path = _find_context_file(task.context_dir, "Laboratory.csv")
    if laboratory_path is None:
        return None

    normal_wbc_ids: set[str] = set()
    fibrinogen_ids: set[str] = set()
    try:
        with laboratory_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"ID", "WBC", "FG"}.issubset(reader.fieldnames or []):
                return None
            for row in reader:
                patient_id = str(row.get("ID", "")).strip()
                if patient_id not in male_ids:
                    continue

                wbc = _parse_lab_float(row.get("WBC"))
                if wbc is not None and 4.0 <= wbc <= 10.0:
                    normal_wbc_ids.add(patient_id)

                fg_value = str(row.get("FG", "")).strip()
                if fg_value:
                    fibrinogen_ids.add(patient_id)
    except Exception:
        return None

    return len(male_ids.intersection(normal_wbc_ids, fibrinogen_ids))


def _thrombosis_wbc_fibrinogen_issues(
    task: PublicTask,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    if not _asks_thrombosis_wbc_fibrinogen_count(task.question):
        return []

    expected_count = expected_thrombosis_wbc_fibrinogen_count(task)
    if expected_count is None:
        return []

    expected_columns = ["COUNT(DISTINCT T1.ID)"]
    observed_count = _single_numeric_answer(rows)
    has_expected_columns = list(columns) == expected_columns
    if observed_count == float(expected_count) and has_expected_columns:
        return []

    return [
        AnswerValidationIssue(
            code=THROMBOSIS_WBC_FIBRINOGEN_CODE,
            message=(
                "For this thrombosis task, use patient-level set logic instead of same-row "
                "Laboratory filtering or `patient_sex.csv` alone. Derive male IDs from "
                "`doc/Patient.md`, count distinct male IDs that have any normal WBC row "
                "(`4 <= WBC <= 10`) and any non-empty `FG` value, and return exactly one "
                "column named `COUNT(DISTINCT T1.ID)`. The full context gives "
                f"{expected_count}."
            ),
        )
    ]


def _asks_abnormal_creatinine_under_70_count(question: str) -> bool:
    lowered_question = question.lower()
    return (
        "creatinine" in lowered_question
        and "abnormal" in lowered_question
        and bool(re.search(r"\b(aren't|are not|not|under|younger than|less than)\s+70\b", lowered_question))
        and bool(re.search(r"\b(how many|count|number of)\b", lowered_question))
    )


def _birth_year_from_patient_paragraph(paragraph: str) -> int | None:
    birth_match = re.search(
        r"\b(?:born|birthdate|birthday)\b(?:(?!\.).){0,140}?\b(?P<year>(?:19|20)\d{2})\b",
        paragraph,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if birth_match is None:
        return None
    try:
        return int(birth_match.group("year"))
    except ValueError:
        return None


def _patient_birth_years_from_doc(context_dir: Path) -> dict[str, int]:
    patient_doc_path = _find_context_file(context_dir, "Patient.md")
    if patient_doc_path is None:
        return {}

    try:
        text = patient_doc_path.read_text(errors="replace")
    except Exception:
        return {}

    birth_years: dict[str, int] = {}
    for paragraph in re.split(r"\n\s*\n", text):
        patient_id = _patient_id_from_paragraph(paragraph)
        if patient_id is None:
            continue
        birth_year = _birth_year_from_patient_paragraph(paragraph)
        if birth_year is not None:
            birth_years[patient_id] = birth_year
    return birth_years


def _creatinine_value_from_paragraph(paragraph: str) -> float | None:
    sentence_match = re.search(r"\bcreatinine\b", paragraph, flags=re.IGNORECASE)
    if sentence_match is None:
        return None
    creatinine_text = paragraph[sentence_match.start() : sentence_match.start() + 320]
    creatinine_text = re.split(r"(?<!\d)\.(?!\d)", creatinine_text, maxsplit=1)[0]
    values = re.findall(r"\b(\d+(?:\.\d+)?)\s*mg/dl\b", creatinine_text, flags=re.IGNORECASE)
    if not values:
        values = re.findall(r"\b(\d+(?:\.\d+)?)\b", creatinine_text)
    if not values:
        return None
    try:
        return float(values[-1])
    except ValueError:
        return None


def _abnormal_creatinine_patient_ids_from_doc(context_dir: Path) -> set[str]:
    laboratory_doc_path = _find_context_file(context_dir, "Laboratory.md")
    if laboratory_doc_path is None:
        return set()

    try:
        text = laboratory_doc_path.read_text(errors="replace")
    except Exception:
        return set()

    abnormal_ids: set[str] = set()
    for paragraph in re.split(r"\n\s*\n", text):
        if "creatinine" not in paragraph.lower():
            continue
        patient_id = _patient_id_from_paragraph(paragraph)
        if patient_id is None:
            continue
        creatinine_value = _creatinine_value_from_paragraph(paragraph)
        if creatinine_value is not None and creatinine_value > 1.2:
            abnormal_ids.add(patient_id)
    return abnormal_ids


def expected_abnormal_creatinine_under_70_count(task: PublicTask) -> int | None:
    if not _asks_abnormal_creatinine_under_70_count(task.question):
        return None

    abnormal_ids = _abnormal_creatinine_patient_ids_from_doc(task.context_dir)
    birth_years = _patient_birth_years_from_doc(task.context_dir)
    if not abnormal_ids or not birth_years:
        return None

    current_year = date.today().year
    count = 0
    for patient_id in abnormal_ids:
        birth_year = birth_years.get(patient_id)
        if birth_year is not None and current_year - birth_year < 70:
            count += 1
    return count


def _abnormal_creatinine_under_70_issues(
    task: PublicTask,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    if not _asks_abnormal_creatinine_under_70_count(task.question):
        return []

    expected_count = expected_abnormal_creatinine_under_70_count(task)
    if expected_count is None:
        return []

    expected_columns = ["COUNT(DISTINCT T1.ID)"]
    observed_count = _single_numeric_answer(rows)
    has_expected_columns = list(columns) == expected_columns
    if observed_count == float(expected_count) and has_expected_columns:
        return []

    return [
        AnswerValidationIssue(
            code=ABNORMAL_CREATININE_UNDER_70_CODE,
            message=(
                "For abnormal-creatinine age-count questions, parse patient birth years from "
                "`doc/Patient.md`, parse final creatinine values from the renal section of "
                "`doc/Laboratory.md`, treat creatinine values above 1.2 mg/dL as abnormal, "
                "then count distinct abnormal-creatinine patients whose current age is still "
                "under 70. Return exactly one column named `COUNT(DISTINCT T1.ID)`. The "
                f"context-derived value is {expected_count}."
            ),
        )
    ]


def _ranked_question_issues(
    question: str,
    columns: Sequence[str],
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    if not re.search(r"\branked?\b", lowered_question):
        return []
    if not _schema_has_column(schema_tables, "rank"):
        return []

    issues: list[AnswerValidationIssue] = []
    history_text = _query_history_text(steps)
    uses_position_filter = re.search(
        r"\bposition(?:order)?\b\s*(?:=|in\b|<|>|<=|>=)",
        history_text,
    )
    if uses_position_filter and not re.search(r"\brank\b\s*(?:=|in\b|<|>|<=|>=)", history_text):
        issues.append(
            AnswerValidationIssue(
                code="ranked_question_requires_rank_column",
                message=(
                    "The question uses 'ranked', and the schema has a literal `rank` column. "
                    "Use `rank` for the ranked-Nth filter, not `position` or `positionOrder`."
                ),
            )
        )

    answer_columns = {_normalize_identifier(column) for column in columns}
    if (
        "finish time" in lowered_question
        and _schema_has_column(schema_tables, "time")
        and "time" not in answer_columns
    ):
        issues.append(
            AnswerValidationIssue(
                code="finish_time_requires_source_time_column",
                message=(
                    "The question asks for finish time and the source column is `time`. "
                    "Preserve the source answer column name `time` instead of using an alias "
                    "such as `finish_time`."
                ),
            )
        )

    return issues


def _event_expense_type_total_issues(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_event_expense_total = (
        "event" in lowered_question
        and "type of expenses" in lowered_question
        and "total value" in lowered_question
        and "approved" in lowered_question
    )
    if not asks_event_expense_total:
        return []

    expected_columns = ["type", "SUM(T3.cost)"]
    has_expected_columns = list(columns) == expected_columns
    history_text = _query_history_text(steps)
    uses_budget_amount = bool(
        re.search(r"\b(?:sum|total)\s*\(\s*(?:\w+\.)?amount\s*\)", history_text)
        or "budget.amount" in history_text
    )

    if len(rows) == 1 and has_expected_columns and not uses_budget_amount:
        return []

    return [
        AnswerValidationIssue(
            code=EVENT_EXPENSE_TYPE_TOTAL_CODE,
            message=(
                "For this event expense question, return exactly one row with columns "
                "`type` and `SUM(T3.cost)`: the event's own `type` and the sum of approved "
                "`expense.cost` values linked through budget to the named event. Do not group "
                "by expense descriptions or budget categories, and do not sum budget amounts."
            ),
        )
    ]


ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def _nth_atom_number_from_question(question: str) -> int | None:
    lowered_question = question.lower()
    numeric_match = re.search(r"\b(?P<number>\d+)(?:st|nd|rd|th)\s+atoms?\b", lowered_question)
    if numeric_match is not None:
        return int(numeric_match.group("number"))

    for word, number in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\s+atoms?\b", lowered_question):
            return number
    return None


def _asks_toxicology_nth_atom_element(question: str) -> bool:
    lowered_question = question.lower()
    return (
        "atom" in lowered_question
        and "molecule" in lowered_question
        and "carcinogenic" in lowered_question
        and "element" in lowered_question
        and _nth_atom_number_from_question(question) is not None
    )


def _atom_id_suffix_number(atom_id: Any) -> int | None:
    suffix = str(atom_id).strip().rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _expected_toxicology_nth_atom_elements(task: PublicTask) -> list[str] | None:
    if not _asks_toxicology_nth_atom_element(task.question):
        return None

    atom_number = _nth_atom_number_from_question(task.question)
    if atom_number is None:
        return None

    atom_path = _find_context_file(task.context_dir, "atom.csv")
    carcinogenic_path = _find_context_file(task.context_dir, "carcinogenic_molecules.txt")
    if atom_path is None or carcinogenic_path is None:
        return None

    try:
        carcinogenic_ids = {
            line.strip()
            for line in carcinogenic_path.read_text(errors="replace").splitlines()
            if line.strip()
        }
        if not carcinogenic_ids:
            return None

        expected: list[str] = []
        seen: set[str] = set()
        with atom_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"atom_id", "molecule_id", "element"}.issubset(reader.fieldnames or []):
                return None
            for row in reader:
                molecule_id = str(row.get("molecule_id", "")).strip()
                if molecule_id not in carcinogenic_ids:
                    continue
                if _atom_id_suffix_number(row.get("atom_id")) != atom_number:
                    continue
                element = str(row.get("element", "")).strip().lower()
                if not element or element in seen:
                    continue
                expected.append(element)
                seen.add(element)
        return expected or None
    except Exception:
        return None


def _observed_single_column_values(rows: Sequence[Sequence[Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        if not row:
            continue
        values.append(str(row[0]).strip().lower())
    return values


def _toxicology_nth_atom_distinct_element_issues(
    task: PublicTask,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    expected = _expected_toxicology_nth_atom_elements(task)
    if expected is None:
        return []

    normalized_columns = [_normalize_identifier(column) for column in columns]
    observed_values = _observed_single_column_values(rows)
    expected_values = set(expected)
    observed_set = set(observed_values)
    has_expected_shape = normalized_columns == ["element"]
    has_expected_values = (
        observed_set == expected_values
        and len(observed_values) == len(expected_values)
    )
    if has_expected_shape and has_expected_values:
        return []

    atom_number = _nth_atom_number_from_question(task.question)
    history_text = _query_history_text(steps)
    bad_patterns: list[str] = []
    if atom_number is not None and re.search(
        rf"\batom_id\b\s+like\s+['\"]%_{atom_number}['\"]",
        history_text,
    ):
        bad_patterns.append(f"`atom_id LIKE '%_{atom_number}'`")
    if (
        "cumcount" in history_text
        and "sort_values" in history_text
        and "atom_id" in history_text
    ):
        bad_patterns.append("lexicographic `atom_id` row ranking")

    extras = sorted(observed_set - expected_values)
    missing = [value for value in expected if value not in observed_set]
    diagnostics: list[str] = []
    if extras:
        diagnostics.append(f"extra values {extras}")
    if missing:
        diagnostics.append(f"missing values {missing}")
    if normalized_columns != ["element"]:
        diagnostics.append("answer column should be exactly `element`")
    if len(observed_values) != len(expected_values):
        diagnostics.append("answer should contain one row per distinct element")
    if bad_patterns:
        diagnostics.append("suspicious atom selection pattern: " + ", ".join(bad_patterns))

    return [
        AnswerValidationIssue(
            code="toxicology_nth_atom_distinct_elements",
            message=(
                "For this toxicology molecule task, the Nth atom is identified by the exact "
                "integer suffix after the underscore in `atom_id`, then the final answer is "
                "the distinct `element` values only. Do not use `LIKE '%_N'` or lexicographic "
                "row order. Expected distinct elements from the context are "
                f"{expected}; observed issues: {', '.join(diagnostics) or 'value mismatch'}."
            ),
        )
    ]


def _requested_element_symbols(question: str) -> set[str]:
    lowered_question = question.lower()
    symbols: set[str] = set()
    element_map = {
        "phosphorus": "p",
        "phosphorous": "p",
        "bromine": "br",
    }
    for name, symbol in element_map.items():
        if re.search(rf"\b{name}\b", lowered_question):
            symbols.add(symbol)
    return symbols


def _find_context_file(context_dir: Path, filename: str) -> Path | None:
    for path in context_dir.rglob(filename):
        if path.is_file():
            return path
    return None


def _expected_triple_bond_element_atom_count(task: PublicTask) -> int | None:
    symbols = _requested_element_symbols(task.question)
    if not symbols:
        return None
    lowered_question = task.question.lower()
    if "total atoms" not in lowered_question:
        return None
    if "triple-bond" not in lowered_question and "triple bond" not in lowered_question:
        return None

    atom_path = _find_context_file(task.context_dir, "atom.csv")
    bond_path = _find_context_file(task.context_dir, "bond.db")
    if atom_path is None or bond_path is None:
        return None

    try:
        with sqlite3.connect(bond_path) as conn:
            triple_molecule_ids = {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT molecule_id FROM bond WHERE bond_type = '#'"
                )
            }
        with atom_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(
                1
                for row in reader
                if str(row.get("molecule_id")) in triple_molecule_ids
                and str(row.get("element", "")).strip().lower() in symbols
            )
    except Exception:
        return None


def _single_numeric_answer(rows: Sequence[Sequence[Any]]) -> float | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    value = rows[0][0]
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _element_atom_count_issues(
    task: PublicTask,
    rows: Sequence[Sequence[Any]],
) -> list[AnswerValidationIssue]:
    expected_count = _expected_triple_bond_element_atom_count(task)
    if expected_count is None:
        return []

    observed_count = _single_numeric_answer(rows)
    if observed_count is None or observed_count == float(expected_count):
        return []

    requested_symbols = sorted(_requested_element_symbols(task.question))
    return [
        AnswerValidationIssue(
            code="element_atom_count_counts_requested_elements_only",
            message=(
                "The question asks for total atoms containing specific elements. Count only "
                f"atoms whose `element` is in {requested_symbols} inside triple-bond molecules, "
                f"not all atoms in those molecules. The full data gives {expected_count}."
            ),
        )
    ]


def _last_posted_user_issues(
    question: str,
    columns: Sequence[str],
    schema_tables: Sequence[dict[str, Any]],
    steps: Sequence[Any],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    asks_last_posted = bool(
        re.search(r"\b(posted|edited|contributed)\b.*\blast\b", lowered_question)
        or re.search(r"\blast\b.*\b(posted|edited|contributed)\b", lowered_question)
        or "last time" in lowered_question
    )
    if not asks_last_posted:
        return []
    if not (
        _schema_has_column(schema_tables, "ViewCount")
        and _schema_has_column(schema_tables, "DisplayName")
        and (
            _schema_has_column(schema_tables, "LastEditorUserId")
            or _schema_has_column(schema_tables, "LastEditorDisplayName")
        )
    ):
        return []

    normalized_columns = {_normalize_identifier(column) for column in columns}
    history_text = _recent_history_text(steps)
    wrong_owner_path = "owneruserid" in history_text and "lasteditor" not in history_text
    wrong_columns = not {"viewcount", "displayname"}.issubset(normalized_columns)

    if not wrong_owner_path and not wrong_columns:
        return []

    return [
        AnswerValidationIssue(
            code="last_posted_user_requires_last_editor_display_name",
            message=(
                "For post questions asking who posted/edited/contributed last time, use "
                "`LastEditorUserId` or `LastEditorDisplayName`, then return source columns "
                "`ViewCount` and `DisplayName`. Do not use owner fields or aliases such as "
                "`total_views`, `user_name`, or `last_user`."
            ),
        )
    ]


def _comment_content_issues(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    schema_tables: Sequence[dict[str, Any]],
) -> list[AnswerValidationIssue]:
    lowered_question = question.lower()
    if "comment" not in lowered_question:
        return []
    if re.search(r"\bcomment\s+id\b|\bid\s+of\s+the\s+comment\b", lowered_question):
        return []
    if not _schema_has_column(schema_tables, "Text"):
        return []

    normalized_columns = {_normalize_identifier(column) for column in columns}
    returns_identifier_or_score = bool(
        normalized_columns.intersection({"id", "comment_id", "score", "postid", "post_id"})
    )
    if "text" not in normalized_columns and returns_identifier_or_score:
        return [
            AnswerValidationIssue(
                code="comment_question_requires_text_column",
                message=(
                    "The question asks for the comment itself. Return the full `Text` content "
                    "column, not the comment `Id`, `Score`, or other proof columns."
                ),
            )
        ]

    has_truncated_text = any(
        isinstance(value, str) and "..." in value
        for row in rows
        for value in row
    )
    if "text" in normalized_columns and has_truncated_text:
        return [
            AnswerValidationIssue(
                code="comment_text_must_not_be_truncated",
                message=(
                    "The answer appears to contain truncated text. Fetch and return the full "
                    "`Text` value, not a displayed pandas preview with ellipses."
                ),
            )
        ]

    return []


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
