from __future__ import annotations

import csv
import json
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _safe_table_name(name: str) -> str:
    normalized = re.sub(r"\W+", "_", name).strip("_").lower()
    if not normalized:
        normalized = "table"
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    return normalized


def _unique_table_name(base_name: str, used_names: set[str]) -> str:
    table_name = _safe_table_name(base_name)
    if table_name not in used_names:
        used_names.add(table_name)
        return table_name

    suffix = 2
    while f"{table_name}_{suffix}" in used_names:
        suffix += 1
    unique_name = f"{table_name}_{suffix}"
    used_names.add(unique_name)
    return unique_name


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _load_json_records(path: Path) -> tuple[str, list[dict[str, Any]]] | None:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        table_name = str(payload.get("table") or path.stem)
        records = payload["records"]
    elif isinstance(payload, list):
        table_name = path.stem
        records = payload
    else:
        return None

    if not all(isinstance(record, dict) for record in records):
        return None
    return table_name, records


def _csv_columns_and_count(path: Path) -> tuple[list[str], int]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return [], 0
    return rows[0], max(len(rows) - 1, 0)


def _json_columns(records: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def _discover_csv_json_tables(context_root: Path) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for path in sorted(context_root.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(context_root).as_posix()
        if path.suffix.lower() == ".csv":
            columns, row_count = _csv_columns_and_count(path)
            table_name = _unique_table_name(path.stem, used_names)
            tables.append(
                {
                    "table": table_name,
                    "source_path": relative_path,
                    "source_type": "csv",
                    "columns": columns,
                    "row_count": row_count,
                }
            )
        elif path.suffix.lower() == ".json":
            loaded = _load_json_records(path)
            if loaded is None:
                continue
            base_name, records = loaded
            table_name = _unique_table_name(base_name, used_names)
            tables.append(
                {
                    "table": table_name,
                    "source_path": relative_path,
                    "source_type": "json",
                    "columns": _json_columns(records),
                    "row_count": len(records),
                    "_records": records,
                }
            )

    return tables


def inspect_context_tables(context_root: Path) -> dict[str, Any]:
    tables = _discover_csv_json_tables(context_root)
    return {
        "tables": [
            {key: value for key, value in table.items() if not key.startswith("_")}
            for table in tables
        ]
    }


def _connect_with_context_tables(context_root: Path) -> tuple[duckdb.DuckDBPyConnection, list[dict[str, Any]]]:
    conn = duckdb.connect(database=":memory:")
    tables = _discover_csv_json_tables(context_root)
    for table in tables:
        quoted_table = _quote_identifier(str(table["table"]))
        source_path = context_root / str(table["source_path"])
        if table["source_type"] == "csv":
            escaped_path = source_path.as_posix().replace("'", "''")
            conn.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_csv_auto('{escaped_path}', header=true)"
            )
        elif table["source_type"] == "json":
            frame = pd.DataFrame(table["_records"])
            conn.register(str(table["table"]), frame)
    return conn, tables


def execute_context_duckdb_sql(context_root: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    normalized_sql = sql.strip()
    if normalized_sql.endswith(";"):
        normalized_sql = normalized_sql[:-1].strip()
    if ";" in normalized_sql:
        raise ValueError("Only one read-only SQL statement is allowed.")
    if not normalized_sql.lower().startswith(("select", "with")):
        raise ValueError("Only SELECT or WITH queries are allowed.")

    conn, tables = _connect_with_context_tables(context_root)
    try:
        cursor = conn.execute(normalized_sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)
    finally:
        conn.close()

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "columns": column_names,
        "rows": [[_jsonable(value) for value in row] for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
        "available_tables": [
            {
                "table": table["table"],
                "source_path": table["source_path"],
                "source_type": table["source_type"],
            }
            for table in tables
        ],
    }
