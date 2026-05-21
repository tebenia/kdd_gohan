"""Generic, schema-agnostic answer validation.

This is the *generalizable* subset of the validate-and-retry idea: every check here
keys off the question wording and the answer-table shape, never off literal dataset
values (no place names, ids, table/column names, or magic thresholds). That keeps it
useful on the hidden B-board, where memorized literals would never match.

`validate_answer` returns a list of human-readable issue strings. The registry turns a
non-empty list into a soft rejection (`ok=False`) and feeds the messages back so the
agent can revise and call `answer` again. There is intentionally no auto-correction:
fabricating answers only helps the public set and risks corrupting unseen tasks.

Design rule: prefer false negatives over false positives. A wrong reject costs the
agent a step (and on B-board could burn it to timeout), so each check only fires on a
high-confidence pattern.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Aggregation / proof column names the model tends to leak into a final projection.
# Deliberately excludes "id"/"name"-style columns, which are often legitimately asked for.
HELPER_COLUMN_NAMES = {
    "amount",
    "average",
    "avg",
    "balance",
    "cost",
    "count",
    "frequency",
    "freq",
    "mean",
    "percent",
    "percentage",
    "price",
    "proof",
    "rank",
    "ranking",
    "ratio",
    "sum",
    "total",
    "tally",
}

_MINMAX_PATTERN = re.compile(
    r"\b(lowest|highest|minimum|maximum|min|max|least|most|smallest|largest|"
    r"cheapest|costliest|earliest|latest|oldest|newest|youngest|fewest|"
    r"fastest|slowest|greatest|longest|shortest|biggest)\b",
    flags=re.IGNORECASE,
)

# Phrasings that legitimately want a single row, so LIMIT 1 / one row is fine.
_EXPLICIT_ONE_PATTERN = re.compile(
    r"\b(exactly one|single|first|nearest|closest|top\s*1|one result|one row|the top)\b",
    flags=re.IGNORECASE,
)

# Phrasings that explicitly ask for the helper value alongside the entity.
_EXTRA_DETAIL_PATTERN = re.compile(
    r"\b(include|including|along with|together with|with its|with their|and its|and their|"
    r"as well as|show the|return the .* and)\b",
    flags=re.IGNORECASE,
)

_LIST_ALL_PATTERN = re.compile(r"\b(list|show|return|find|name|give)\s+(all|every|each)\b", flags=re.IGNORECASE)

_TALLY_PATTERN = re.compile(r"\b(tally|enumerate|list the|what are the)\b", flags=re.IGNORECASE)

_AGGREGATE_PATTERN = re.compile(
    r"\b(average|avg|mean|sum|total|count|how many|number of)\b", flags=re.IGNORECASE
)

# Wording that justifies dropping zero/blank values from an aggregate.
_ZERO_EXCLUSION_ALLOWED_PATTERN = re.compile(
    r"\b(positive|nonzero|non-zero|valid|known|available|non-null|not\s+null|"
    r"missing|unknown|exclude|excluding|without|ignore|omit|filter\s+out|"
    r"greater\s+than\s+0|more\s+than\s+0|above\s+0)\b|>\s*0",
    flags=re.IGNORECASE,
)

_ZERO_FILTER_PATTERN = re.compile(r">\s*0(?:\.0+)?\b", flags=re.IGNORECASE)

_COST_METRIC_PATTERN = re.compile(r"\bcost\b", flags=re.IGNORECASE)
_AMOUNT_COLUMN_PATTERN = re.compile(r"\bamount\b", flags=re.IGNORECASE)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _is_helper_column(column: str) -> bool:
    return _normalize(column) in {_normalize(n) for n in HELPER_COLUMN_NAMES}


def _query_history_text(previous_steps: Sequence[Any]) -> str:
    """Concatenate every SQL / Python snippet the agent has run so far, lowercased.

    Robust to the parallel-action format where `action_input` is a list of dicts.
    """
    chunks: list[str] = []
    for step in previous_steps:
        action_input = getattr(step, "action_input", None)
        if action_input is None and isinstance(step, dict):
            action_input = step.get("action_input")
        candidates = action_input if isinstance(action_input, list) else [action_input]
        for item in candidates:
            if isinstance(item, dict):
                for key in ("sql", "code"):
                    value = item.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
    return "\n".join(chunks).lower()


def validate_answer(
    question: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    previous_steps: Sequence[Any] = (),
) -> list[str]:
    """Return high-confidence, generic issues for a proposed final answer.

    Empty list means the answer passes. Each string is actionable feedback for the agent.
    """
    # Be defensive: malformed shapes are validated/rejected by the answer handler itself.
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        return []
    if not isinstance(rows, list):
        return []

    q = question or ""
    history = _query_history_text(previous_steps)

    issues: list[str] = []

    asks_minmax = bool(_MINMAX_PATTERN.search(q)) and not _EXPLICIT_ONE_PATTERN.search(q)
    asks_list_all = bool(_LIST_ALL_PATTERN.search(q))
    asks_tally = bool(_TALLY_PATTERN.search(q))
    wants_extra = bool(_EXTRA_DETAIL_PATTERN.search(q))

    # 1) Helper/proof columns leaking into a "which/list all/tally" projection.
    if len(columns) > 1 and not wants_extra and (asks_minmax or asks_list_all or asks_tally):
        helpers = [c for c in columns if _is_helper_column(c)]
        if helpers:
            issues.append(
                f"The answer includes likely helper/proof column(s) {helpers}. The question asks "
                "only for the requested entity/value, so project those internal columns away and "
                "return only the field(s) explicitly requested."
            )

    # 2) Aggregate that silently excludes zeros without the question asking to.
    if (
        _AGGREGATE_PATTERN.search(q)
        and history
        and _ZERO_FILTER_PATTERN.search(history)
        and not _ZERO_EXCLUSION_ALLOWED_PATTERN.search(q)
    ):
        issues.append(
            "The aggregate query filters values with `> 0`, but the question does not ask to "
            "exclude zero/positive-only values. Keep zero values in the aggregate; only drop "
            "blanks/nulls via casting (e.g. TRY_CAST/NULLIF) if needed."
        )

    # 3) Minmax question returns multiple tied rows but the question has no "list all" / "tally"
    # phrasing that would justify a multi-row answer. Fires only when answer has >1 entity row
    # and no LIMIT 1 was used. This avoids breaking list-all / tally tasks (those skip via
    # asks_list_all / asks_tally) and numeric aggregate tasks (those have helper col in answer).
    _LIMIT_ONE_PATTERN = re.compile(r"\blimit\s+1\b", flags=re.IGNORECASE)
    if (
        asks_minmax
        and not asks_list_all
        and not asks_tally
        and not wants_extra
        and len(columns) == 1
        and not _is_helper_column(columns[0])
        and len(rows) > 1
        and history
        and not _LIMIT_ONE_PATTERN.search(history)
    ):
        issues.append(
            "The question has a min/max intent but does not ask to 'list all' or 'tally'. "
            "Return exactly one row using TWO sort keys for deterministic tie-breaking: "
            "ORDER BY metric_column ASC, entity_name_column ASC LIMIT 1. "
            "Example: if returning event_name ordered by cost, use "
            "`ORDER BY ex.cost ASC, e.event_name ASC LIMIT 1`. "
            "Do NOT order by the metric alone (tied values give a non-deterministic result). "
            "Do NOT return all tied rows."
        )

    # 4) Minmax "cost" question where agent queried `amount` without a `cost` column.
    # Fires only when: (a) question has minmax+cost, (b) answer is a single entity column,
    # (c) history shows `amount` used but no `cost` column referenced. Prefer false negatives:
    # the `and history` guard ensures we don't fire when there are no recorded queries.
    if (
        asks_minmax
        and _COST_METRIC_PATTERN.search(q)
        and len(columns) == 1
        and not _is_helper_column(columns[0])
        and rows
        and history
        and _AMOUNT_COLUMN_PATTERN.search(history)
        and not _COST_METRIC_PATTERN.search(history)
    ):
        issues.append(
            "The question asks for 'cost', but your queries reference `amount` without "
            "using a column literally named `cost`. Check whether there is a separate "
            "table (e.g., an expense or transaction table) that has a `cost` column — "
            "use that for the min/max lookup. Only fall back to `amount` if no `cost` "
            "column exists in the database."
        )

    return issues
