from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.answer_validator import validate_answer


def _task(tmp_path: Path, question: str) -> PublicTask:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    return PublicTask(
        record=TaskRecord(task_id="task_test", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_records(path: Path, table: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"table": table, "records": rows}))


def _step(action: str, action_input: dict[str, object], ok: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        action=action,
        action_input=action_input,
        observation={"ok": ok},
        raw_response=json.dumps({"action": action, "action_input": action_input}),
    )


def _issue_codes(issues) -> set[str]:
    return {issue.code for issue in issues}


class AnswerValidatorTests(unittest.TestCase):
    def test_rejects_extra_helper_column_for_lowest_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Which event has the lowest cost?")

            issues = validate_answer(
                task,
                columns=["event_name", "cost"],
                rows=[["November Speaker", 10]],
            )

        self.assertIn("extra_helper_columns", _issue_codes(issues))

    def test_rejects_limit_one_for_minmax_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Which event has the lowest cost?")
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT event_name FROM events ORDER BY cost ASC LIMIT 1"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["event_name"],
                rows=[["October Speaker"]],
                previous_steps=previous_steps,
            )

        self.assertIn("minmax_limit_one", _issue_codes(issues))

    def test_rejects_driver_number_without_drivers_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "What is his number of the driver who finished 0:01:54 in Q3?",
            )
            _write_csv(
                task.context_dir / "qualifying.csv",
                [{"raceId": 903, "driverId": 20, "number": 1, "q3": "1:54.960"}],
            )
            _write_json_records(
                task.context_dir / "drivers.json",
                "drivers",
                [{"driverId": 20, "number": 5, "forename": "Sebastian", "surname": "Vettel"}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT number FROM qualifying WHERE raceId = 903 AND q3 LIKE '1:54%'"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["number"],
                rows=[[1]],
                previous_steps=previous_steps,
            )

        self.assertIn("driver_number_requires_drivers_table", _issue_codes(issues))

    def test_accepts_driver_number_with_drivers_table_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "What is his number of the driver who finished 0:01:54 in Q3?",
            )
            _write_csv(
                task.context_dir / "qualifying.csv",
                [{"raceId": 903, "driverId": 20, "number": 1, "q3": "1:54.960"}],
            )
            _write_json_records(
                task.context_dir / "drivers.json",
                "drivers",
                [{"driverId": 20, "number": 5, "forename": "Sebastian", "surname": "Vettel"}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT d.number FROM qualifying q "
                            "JOIN drivers d ON q.driverId = d.driverId "
                            "WHERE q.raceId = 903 AND q.q3 LIKE '1:54%'"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["number"],
                rows=[[5]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("driver_number_requires_drivers_table", _issue_codes(issues))

    def test_rejects_merged_full_name_when_source_has_split_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "List the full names of employees.")
            _write_csv(
                task.context_dir / "people.csv",
                [{"first_name": "Ada", "last_name": "Lovelace"}],
            )

            issues = validate_answer(
                task,
                columns=["full_name"],
                rows=[["Ada Lovelace"]],
            )

        self.assertIn("merged_name_columns", _issue_codes(issues))


if __name__ == "__main__":
    unittest.main()
