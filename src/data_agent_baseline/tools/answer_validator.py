from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
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
    issues.extend(_consumption_status_projection_issues(task.question, columns))
    issues.extend(_merged_name_issues(columns, schema_tables))
    issues.extend(_minmax_limit_issues(task.question, validation_steps))
    issues.extend(_event_lowest_cost_sum_issues(task.question, schema_tables, validation_steps))
    issues.extend(_aggregate_zero_exclusion_issues(task.question, validation_steps))
    issues.extend(_per_unit_price_issues(task.question, schema_tables, validation_steps))
    issues.extend(_empty_gas_station_country_issues(task, columns, rows, schema_tables))
    issues.extend(_california_schools_sat_issues(task.question, validation_steps))
    issues.extend(_finance_cash_withdrawal_issues(task.question, validation_steps))
    issues.extend(_formula1_track_number_issues(task.question, validation_steps))
    issues.extend(_ranked_question_issues(task.question, columns, schema_tables, validation_steps))
    issues.extend(_event_expense_type_total_issues(task.question, columns, rows, validation_steps))
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
    for pattern in patterns:
        for match in re.finditer(pattern, history_text, flags=re.IGNORECASE):
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
            code="event_expense_type_total_requires_event_type_and_expense_cost_sum",
            message=(
                "For this event expense question, return exactly one row with columns "
                "`type` and `SUM(T3.cost)`: the event's own `type` and the sum of approved "
                "`expense.cost` values linked through budget to the named event. Do not group "
                "by expense descriptions or budget categories, and do not sum budget amounts."
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
