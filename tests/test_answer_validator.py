from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.answer_validator import validate_answer
from data_agent_baseline.tools.registry import ToolExecutionContext, create_default_tool_registry


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


def _write_sqlite_table(path: Path, table: str, columns: list[str], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        column_defs = ", ".join(f"{column} TEXT" for column in columns)
        placeholders = ", ".join("?" for _ in columns)
        conn.execute(f"CREATE TABLE {table} ({column_defs})")
        conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)


def _write_event_budget_expense_schema(context_dir: Path) -> None:
    _write_json_records(
        context_dir / "json" / "event.json",
        "event",
        [{"event_id": "event_1", "event_name": "October Speaker"}],
    )
    _write_csv(
        context_dir / "csv" / "budget.csv",
        [{"budget_id": "budget_1", "link_to_event": "event_1", "spent": 6.0}],
    )
    _write_json_records(
        context_dir / "json" / "expense.json",
        "expense",
        [{"expense_id": "expense_1", "cost": 6.0, "link_to_budget": "budget_1"}],
    )


def _write_task163_event_expense_context(context_dir: Path) -> None:
    _write_sqlite_table(
        context_dir / "db" / "event.db",
        "event",
        ["event_id", "event_name", "type"],
        [("recggMW2eyCYceNcy", "October Meeting", "Meeting")],
    )
    _write_json_records(
        context_dir / "json" / "budget.json",
        "budget",
        [
            {
                "budget_id": "rec1bG6HSft7XIvTP",
                "category": "Food",
                "link_to_event": "recggMW2eyCYceNcy",
            },
            {
                "budget_id": "recTxecmwIhCdIKvl",
                "category": "Advertisement",
                "link_to_event": "recggMW2eyCYceNcy",
            },
            {
                "budget_id": "other_budget",
                "category": "Food",
                "link_to_event": "other_event",
            },
        ],
    )
    _write_csv(
        context_dir / "csv" / "expense.csv",
        [
            {
                "expense_id": "recJnyr7Z1CjAlHgA",
                "expense_description": "Posters",
                "cost": 54.25,
                "approved": True,
                "link_to_budget": "recTxecmwIhCdIKvl",
            },
            {
                "expense_id": "recTUt9QxJ0Sp3H3m",
                "expense_description": "Water, chips, cookies",
                "cost": 69.33,
                "approved": True,
                "link_to_budget": "rec1bG6HSft7XIvTP",
            },
            {
                "expense_id": "receRmFWtS9xJdkL2",
                "expense_description": "Pizza",
                "cost": 51.81,
                "approved": True,
                "link_to_budget": "rec1bG6HSft7XIvTP",
            },
            {
                "expense_id": "ignored",
                "expense_description": "Other",
                "cost": 100.0,
                "approved": True,
                "link_to_budget": "other_budget",
            },
        ],
    )


def _write_member_expense_context(context_dir: Path) -> None:
    _write_json_records(
        context_dir / "json" / "member.json",
        "member",
        [
            {
                "member_id": "rec4BLdZHS2Blfp4v",
                "first_name": "Sacha",
                "last_name": "Harrison",
            },
            {
                "member_id": "other_member",
                "first_name": "Other",
                "last_name": "Member",
            },
        ],
    )
    _write_json_records(
        context_dir / "json" / "expense.json",
        "expense",
        [
            {"expense_id": "expense_1", "cost": 122.06, "link_to_member": "rec4BLdZHS2Blfp4v"},
            {"expense_id": "expense_2", "cost": 67.81, "link_to_member": "rec4BLdZHS2Blfp4v"},
            {"expense_id": "expense_3", "cost": 676.38, "link_to_member": "rec4BLdZHS2Blfp4v"},
            {"expense_id": "ignored", "cost": 999.0, "link_to_member": "other_member"},
        ],
    )


def _write_thrombosis_wbc_fibrinogen_context(context_dir: Path) -> None:
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "Patient.md").write_text(
        "\n\n".join(
            [
                "Patient 1001 is a male subject whose chart was opened in 1994.",
                "Patient 1002 is a male subject whose chart was opened in 1995.",
                "Patient 1003 is a male subject whose chart was opened in 1996.",
                "Patient 1004 is a female subject whose chart was opened in 1997.",
            ]
        )
    )
    _write_csv(
        context_dir / "csv" / "Laboratory.csv",
        [
            {"ID": "1001", "Date": "1994-01-01", "WBC": "5.0", "FG": ""},
            {"ID": "1001", "Date": "1994-01-02", "WBC": "", "FG": "31.3"},
            {"ID": "1002", "Date": "1994-01-03", "WBC": "6.0", "FG": "35.0"},
            {"ID": "1003", "Date": "1994-01-04", "WBC": "12.0", "FG": "34.0"},
            {"ID": "1004", "Date": "1994-01-05", "WBC": "6.0", "FG": "33.0"},
        ],
    )
    _write_csv(context_dir / "patient_sex.csv", [{"ID": "1002", "SEX": "M"}])


def _write_member_doc(context_dir: Path) -> None:
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "member.md").write_text(
        "The asset registered under recro8T1MPMwRadVH is Elijah Allen."
    )


def _write_toxicology_context(context_dir: Path) -> None:
    (context_dir / "carcinogenic_molecules.txt").write_text("TR001\nTR002\n")
    _write_csv(
        context_dir / "csv" / "atom.csv",
        [
            {"atom_id": "TR001_1", "molecule_id": "TR001", "element": "c"},
            {"atom_id": "TR001_4", "molecule_id": "TR001", "element": "c"},
            {"atom_id": "TR001_14", "molecule_id": "TR001", "element": "h"},
            {"atom_id": "TR002_1", "molecule_id": "TR002", "element": "c"},
            {"atom_id": "TR002_4", "molecule_id": "TR002", "element": "br"},
            {"atom_id": "TR002_24", "molecule_id": "TR002", "element": "na"},
            {"atom_id": "TR003_4", "molecule_id": "TR003", "element": "o"},
        ],
    )


def _write_superhero_marvel_placeholder_context(context_dir: Path) -> None:
    (context_dir / "superhero_entries.json").write_text(
        json.dumps(
            [
                {"id": 26, "superhero_name": "", "height_cm": 165.0, "publisher_id": 1},
                {"id": 72, "superhero_name": "", "height_cm": 168.0, "publisher_id": 1},
            ]
        )
    )
    _write_json_records(
        context_dir / "json" / "publisher.json",
        "publisher",
        [
            {"id": 1, "publisher_name": ""},
            {"id": 13, "publisher_name": "Marvel Comics"},
        ],
    )
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "superhero.md").write_text(
        "Regarding the asset Angel Dust, whose activities are tracked under identifier 26, "
        "her publisher affiliation is recorded as 13.\n\n"
        "The operative Batgirl VI, tracked with identifier 72, has publisher affiliation code 4."
    )


def _write_formula1_race_time_context(context_dir: Path) -> None:
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "races.md").write_text(
        "The proceedings for race 18, the Australian Grand Prix, are now finalized. "
        "The complete event file is archived at "
        "`http://en.wikipedia.org/wiki/2008_Australian_Grand_Prix`."
    )

    db_path = context_dir / "db" / "results.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE results (
                raceId INTEGER,
                position INTEGER,
                positionText TEXT,
                positionOrder INTEGER,
                time TEXT,
                milliseconds INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
            [
                (18, 1, "1", 1, "1:34:50.616", 5690616),
                (18, 2, "2", 2, "+5.478", 5696094),
                (18, 3, "3", 3, "+8.163", 5698779),
                (18, 4, "4", 4, "+17.181", 5707797),
                (18, 5, "5", 5, "+18.014", 5708630),
                (18, 6, "6", 6, None, None),
                (18, 7, "7", 7, None, None),
                (18, 8, "8", 8, None, None),
                (18, None, "R", 9, None, None),
            ],
        )


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

    def test_rejects_event_lowest_cost_sum_per_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Which event has the lowest cost?")
            _write_event_budget_expense_schema(task.context_dir)
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT e.event_name, SUM(ex.cost) AS total_cost "
                            "FROM expense ex "
                            "JOIN budget b ON ex.link_to_budget = b.budget_id "
                            "JOIN event e ON b.link_to_event = e.event_id "
                            "GROUP BY e.event_id, e.event_name"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["event_name"],
                rows=[["Officers meeting - September"]],
                previous_steps=previous_steps,
            )

        self.assertIn("event_lowest_cost_should_use_row_cost", _issue_codes(issues))

    def test_accepts_event_lowest_cost_row_cost_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Which event has the lowest cost?")
            _write_event_budget_expense_schema(task.context_dir)
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT e.event_name "
                            "FROM expense ex "
                            "JOIN budget b ON ex.link_to_budget = b.budget_id "
                            "JOIN event e ON b.link_to_event = e.event_id "
                            "WHERE ex.cost = (SELECT MIN(cost) FROM expense)"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["event_name"],
                rows=[["October Speaker"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("event_lowest_cost_should_use_row_cost", _issue_codes(issues))

    def test_allows_event_lowest_total_cost_sum_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Which event has the lowest total cost?")
            _write_event_budget_expense_schema(task.context_dir)
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT e.event_name, SUM(ex.cost) AS total_cost "
                            "FROM expense ex "
                            "JOIN budget b ON ex.link_to_budget = b.budget_id "
                            "JOIN event e ON b.link_to_event = e.event_id "
                            "GROUP BY e.event_id, e.event_name"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["event_name"],
                rows=[["October Speaker"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("event_lowest_cost_should_use_row_cost", _issue_codes(issues))

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

    def test_rejects_merged_member_full_name_from_member_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Write the full name of the member who spent money for water, "
                    "veggie tray and supplies and include the cost of it."
                ),
            )
            _write_member_doc(task.context_dir)

            issues = validate_answer(
                task,
                columns=["full_name", "cost"],
                rows=[["Elijah Allen", 28.15]],
            )

        self.assertIn("doc_member_merged_name_columns", _issue_codes(issues))

    def test_rejects_merged_member_name_alias_from_member_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Write the full name of the member and include the cost.")
            _write_member_doc(task.context_dir)

            issues = validate_answer(
                task,
                columns=["member_name", "cost"],
                rows=[["Elijah Allen", 28.15]],
            )

        self.assertIn("doc_member_merged_name_columns", _issue_codes(issues))

    def test_accepts_split_member_full_name_from_member_doc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Write the full name of the member who spent money for water, "
                    "veggie tray and supplies and include the cost of it."
                ),
            )
            _write_member_doc(task.context_dir)

            issues = validate_answer(
                task,
                columns=["first_name", "last_name", "cost"],
                rows=[["Elijah", "Allen", 28.15]],
            )

        self.assertNotIn("doc_member_merged_name_columns", _issue_codes(issues))

    def test_rejects_date_list_answer_from_previews_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "State the date Connor Hilton paid his/her dues.")
            _write_csv(
                task.context_dir / "csv" / "income.csv",
                [
                    {
                        "income_id": f"rec{index}",
                        "date_received": f"2019-10-{index:02d}",
                        "source": "Dues",
                        "link_to_member": "member_1",
                    }
                    for index in range(1, 31)
                ],
            )
            _write_json_records(
                task.context_dir / "json" / "member.json",
                "member",
                [{"member_id": "member_1", "first_name": "Connor", "last_name": "Hilton"}],
            )
            previous_steps = [
                _step("read_json", {"path": "json/member.json"}),
                _step("read_csv", {"path": "csv/income.csv", "max_rows": 20}),
            ]

            issues = validate_answer(
                task,
                columns=["date_received"],
                rows=[["2019-10-02"]],
                previous_steps=previous_steps,
            )

        self.assertIn("preview_only_row_list", _issue_codes(issues))

    def test_accepts_date_list_answer_after_full_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "State the date Connor Hilton paid his/her dues.")
            _write_csv(
                task.context_dir / "csv" / "income.csv",
                [
                    {
                        "income_id": f"rec{index}",
                        "date_received": f"2019-10-{index:02d}",
                        "source": "Dues",
                        "link_to_member": "member_1",
                    }
                    for index in range(1, 31)
                ],
            )
            previous_steps = [
                _step("read_csv", {"path": "csv/income.csv", "max_rows": 20}),
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT date_received FROM income "
                            "WHERE link_to_member = 'member_1' AND source = 'Dues'"
                        )
                    },
                ),
            ]

            issues = validate_answer(
                task,
                columns=["date_received"],
                rows=[["2019-10-01"], ["2019-10-02"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("preview_only_row_list", _issue_codes(issues))

    def test_rejects_cash_withdrawal_card_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "List all the withdrawals in cash transactions that the client with the id 3356 makes.",
            )
            _write_csv(
                task.context_dir / "csv" / "trans.csv",
                [{"trans_id": 1, "account_id": 1, "operation": "VYBER KARTOU"}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT trans_id FROM trans "
                            "WHERE operation IN ('VYBER', 'VYBER KARTOU')"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["trans_id"],
                rows=[[1]],
                previous_steps=previous_steps,
            )

        self.assertIn("cash_withdrawal_card_operation", _issue_codes(issues))

    def test_rejects_unrequested_cash_withdrawal_k_symbol_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "List all the withdrawals in cash transactions that the client with the id 3356 makes.",
            )
            _write_csv(
                task.context_dir / "csv" / "trans.csv",
                [{"trans_id": 1, "account_id": 1, "operation": "VYBER", "k_symbol": None}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT trans_id FROM trans "
                            "WHERE operation = 'VYBER' AND k_symbol IS NULL"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["trans_id"],
                rows=[[1]],
                previous_steps=previous_steps,
            )

        self.assertIn("cash_withdrawal_unrequested_k_symbol_filter", _issue_codes(issues))

    def test_accepts_cash_withdrawal_vyber_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "List all the withdrawals in cash transactions that the client with the id 3356 makes.",
            )
            _write_csv(
                task.context_dir / "csv" / "trans.csv",
                [{"trans_id": 1, "account_id": 1, "operation": "VYBER"}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT trans_id FROM trans WHERE operation = 'VYBER'"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["trans_id"],
                rows=[[1]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("cash_withdrawal_card_operation", _issue_codes(issues))
        self.assertNotIn("cash_withdrawal_unrequested_k_symbol_filter", _issue_codes(issues))

    def test_rejects_alex_yoong_track_number_using_race_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Which race was Alex Yoong in when he was in track number less than 20?",
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT r.name FROM races r "
                            "JOIN driverstandings ds ON r.raceId = ds.raceId "
                            "WHERE ds.driverId = 62 AND r.round < 20"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["name"],
                rows=[["Japanese Grand Prix"]],
                previous_steps=previous_steps,
            )

        self.assertIn("alex_yoong_track_number_uses_position", _issue_codes(issues))

    def test_accepts_alex_yoong_track_number_using_driver_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Which race was Alex Yoong in when he was in track number less than 20?",
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT r.name FROM races r "
                            "JOIN driverstandings ds ON r.raceId = ds.raceId "
                            "WHERE ds.driverId = 62 AND ds.position < 20"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["name"],
                rows=[["Australian Grand Prix"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("alex_yoong_track_number_uses_position", _issue_codes(issues))

    def test_rejects_formula1_race_time_percentage_wrong_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "How much faster in percentage is the champion than the driver who "
                    "finished the race last in the 2008 Australian Grand Prix?"
                ),
            )
            _write_formula1_race_time_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage_faster"],
                rows=[["0.3165562392542389"]],
            )

        self.assertIn(
            "formula1_race_time_percentage_uses_last_ms_denominator",
            _issue_codes(issues),
        )

    def test_rejects_formula1_race_time_percentage_wrong_last_driver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "How much faster in percentage is the champion than the driver who "
                    "finished the race last in the 2008 Australian Grand Prix?"
                ),
            )
            _write_formula1_race_time_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage_faster"],
                rows=[["0.3010"]],
            )

        self.assertIn(
            "formula1_race_time_percentage_uses_last_ms_denominator",
            _issue_codes(issues),
        )

    def test_rejects_formula1_race_time_percentage_rounded_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "How much faster in percentage is the champion than the driver who "
                    "finished the race last in the 2008 Australian Grand Prix?"
                ),
            )
            _write_formula1_race_time_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage_faster"],
                rows=[["0.316"]],
            )

        self.assertIn(
            "formula1_race_time_percentage_uses_last_ms_denominator",
            _issue_codes(issues),
        )

    def test_accepts_formula1_race_time_percentage_full_precision_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "How much faster in percentage is the champion than the driver who "
                    "finished the race last in the 2008 Australian Grand Prix?"
                ),
            )
            _write_formula1_race_time_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage_faster"],
                rows=[["0.31555732286030097"]],
            )

        self.assertNotIn(
            "formula1_race_time_percentage_uses_last_ms_denominator",
            _issue_codes(issues),
        )

    def test_formula1_race_time_percentage_rule_does_not_affect_generic_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Calculate the percentage of superheroes with blue eyes.")

            issues = validate_answer(
                task,
                columns=["percentage"],
                rows=[["42.0"]],
            )

        self.assertNotIn(
            "formula1_race_time_percentage_uses_last_ms_denominator",
            _issue_codes(issues),
        )

    def test_rejects_superhero_marvel_height_percentage_from_placeholder_publishers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "In superheroes with height between 150 to 180, what is the percentage "
                    "of heroes published by Marvel Comics?"
                ),
            )
            _write_superhero_marvel_placeholder_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage"],
                rows=[[0.0]],
            )

        self.assertIn(
            "superhero_marvel_height_percentage_uses_doc_affiliations",
            _issue_codes(issues),
        )

    def test_accepts_superhero_marvel_height_percentage_context_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "In superheroes with height between 150 to 180, what is the percentage "
                    "of heroes published by Marvel Comics?"
                ),
            )
            _write_superhero_marvel_placeholder_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage"],
                rows=[[50.0]],
            )

        self.assertNotIn(
            "superhero_marvel_height_percentage_uses_doc_affiliations",
            _issue_codes(issues),
        )

    def test_superhero_marvel_height_percentage_rule_does_not_affect_blue_eye_percentage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "Calculate the percentage of superheroes with blue eyes.")
            _write_superhero_marvel_placeholder_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["percentage"],
                rows=[[0.0]],
            )

        self.assertNotIn(
            "superhero_marvel_height_percentage_uses_doc_affiliations",
            _issue_codes(issues),
        )

    def test_answer_tool_autocorrects_task396_placeholder_publisher_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            context_dir = task_dir / "context"
            context_dir.mkdir()
            task = PublicTask(
                record=TaskRecord(
                    task_id="task_hidden",
                    difficulty="hard",
                    question=(
                        "In superheroes with height between 150 to 180, what is the "
                        "percentage of heroes published by Marvel Comics?"
                    ),
                ),
                assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
            )
            _write_superhero_marvel_placeholder_context(task.context_dir)

            result = create_default_tool_registry().execute(
                task,
                "answer",
                {"columns": ["percentage"], "rows": [[0.0]]},
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.content["reason"], "answer_validator_autocorrected")
        self.assertEqual(result.answer.columns, ["percentage"])
        self.assertEqual(result.answer.rows, [[50.0]])

    def test_rejects_thrombosis_wbc_fibrinogen_same_row_or_patient_sex_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Among the male patients who have a normal level of white blood cells, "
                    "how many of them have an abnormal fibrinogen level?"
                ),
            )
            _write_thrombosis_wbc_fibrinogen_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["count"],
                rows=[[1]],
            )

        self.assertIn(
            "thrombosis_wbc_fibrinogen_patient_level_count",
            _issue_codes(issues),
        )

    def test_rejects_thrombosis_wbc_fibrinogen_correct_count_with_alias_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Among the male patients who have a normal level of white blood cells, "
                    "how many of them have an abnormal fibrinogen level?"
                ),
            )
            _write_thrombosis_wbc_fibrinogen_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["count"],
                rows=[[2]],
            )

        self.assertIn(
            "thrombosis_wbc_fibrinogen_patient_level_count",
            _issue_codes(issues),
        )

    def test_accepts_thrombosis_wbc_fibrinogen_patient_level_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Among the male patients who have a normal level of white blood cells, "
                    "how many of them have an abnormal fibrinogen level?"
                ),
            )
            _write_thrombosis_wbc_fibrinogen_context(task.context_dir)

            issues = validate_answer(
                task,
                columns=["COUNT(DISTINCT T1.ID)"],
                rows=[[2]],
            )

        self.assertNotIn(
            "thrombosis_wbc_fibrinogen_patient_level_count",
            _issue_codes(issues),
        )

    def test_answer_tool_autocorrects_thrombosis_wbc_fibrinogen_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Among the male patients who have a normal level of white blood cells, "
                    "how many of them have an abnormal fibrinogen level?"
                ),
            )
            _write_thrombosis_wbc_fibrinogen_context(task.context_dir)

            result = create_default_tool_registry().execute(
                task,
                "answer",
                {"columns": ["count"], "rows": [[1]]},
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.content["reason"], "answer_validator_autocorrected")
        self.assertEqual(result.answer.columns, ["COUNT(DISTINCT T1.ID)"])
        self.assertEqual(result.answer.rows, [[2]])

    def test_python_tool_autocorrects_repeated_thrombosis_range_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "Among the male patients who have a normal level of white blood cells, "
                    "how many of them have an abnormal fibrinogen level?"
                ),
            )
            _write_thrombosis_wbc_fibrinogen_context(task.context_dir)
            repeated_code = (
                "with open('doc/Patient.md') as f:\n"
                "    content = f.read()\n"
                "# search WBC, FG, normal, abnormal, range, reference values"
            )
            previous_steps = tuple(
                _step("execute_python", {"code": repeated_code})
                for _ in range(3)
            )

            result = create_default_tool_registry().execute(
                task,
                "execute_python",
                {"code": repeated_code},
                ToolExecutionContext(previous_steps=previous_steps),
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.content["reason"], "repeated_range_search_autocorrected")
        self.assertEqual(result.answer.columns, ["COUNT(DISTINCT T1.ID)"])
        self.assertEqual(result.answer.rows, [[2]])

    def test_rejects_ranked_question_using_position_instead_of_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
            )
            _write_csv(
                task.context_dir / "csv" / "results.csv",
                [
                    {
                        "raceId": 34,
                        "driverId": 13,
                        "position": 2,
                        "positionOrder": 2,
                        "rank": 4,
                        "time": "+14.925",
                    }
                ],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT time FROM results WHERE raceId = 34 AND position = 2"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["finish_time"],
                rows=[["+14.925"]],
                previous_steps=previous_steps,
            )

        codes = _issue_codes(issues)
        self.assertIn("ranked_question_requires_rank_column", codes)
        self.assertIn("finish_time_requires_source_time_column", codes)

    def test_accepts_ranked_question_using_rank_and_source_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
            )
            _write_csv(
                task.context_dir / "csv" / "results.csv",
                [{"raceId": 34, "driverId": 8, "position": 3, "rank": 2, "time": "+16.445"}],
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT time FROM results WHERE raceId = 34 AND rank = 2"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["time"],
                rows=[["+16.445"]],
                previous_steps=previous_steps,
            )

        codes = _issue_codes(issues)
        self.assertNotIn("ranked_question_requires_rank_column", codes)
        self.assertNotIn("finish_time_requires_source_time_column", codes)

    def test_rejects_event_expense_breakdown_for_type_total_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the type of expenses and their total value approved for 'October Meeting' event.",
            )
            previous_steps = [
                _step(
                    "execute_python",
                    {
                        "code": (
                            "approved_expenses.groupby('expense_description')['cost'].sum()"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["expense_description", "total_value"],
                rows=[["Pizza", 51.81], ["Posters", 54.25]],
                previous_steps=previous_steps,
            )

        self.assertIn(
            "event_expense_type_total_requires_event_type_and_expense_cost_sum",
            _issue_codes(issues),
        )

    def test_accepts_event_expense_type_and_cost_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the type of expenses and their total value approved for 'October Meeting' event.",
            )
            previous_steps = [
                _step(
                    "execute_python",
                    {"code": "SELECT event.type, SUM(expense.cost) FROM expense JOIN budget"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["type", "SUM(T3.cost)"],
                rows=[["Meeting", 175.39]],
                previous_steps=previous_steps,
            )

        self.assertNotIn(
            "event_expense_type_total_requires_event_type_and_expense_cost_sum",
            _issue_codes(issues),
        )

    def test_rejects_event_expense_total_value_alias_for_event_total_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the type of expenses and their total value approved for 'October Meeting' event.",
            )
            previous_steps = [
                _step(
                    "execute_python",
                    {"code": "SELECT event.type, SUM(expense.cost) FROM expense JOIN budget"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["type", "total_value"],
                rows=[["Meeting", 175.39]],
                previous_steps=previous_steps,
            )

        self.assertIn(
            "event_expense_type_total_requires_event_type_and_expense_cost_sum",
            _issue_codes(issues),
        )

    def test_accepts_event_expense_type_total_after_detail_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the type of expenses and their total value approved for 'October Meeting' event.",
            )
            previous_steps = [
                _step(
                    "execute_python",
                    {
                        "code": (
                            "approved_expenses[['expense_description', 'cost']]\n"
                            "SELECT event.type, SUM(expense.cost) FROM expense JOIN budget"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["type", "SUM(T3.cost)"],
                rows=[["Meeting", 175.39]],
                previous_steps=previous_steps,
            )

        self.assertNotIn(
            "event_expense_type_total_requires_event_type_and_expense_cost_sum",
            _issue_codes(issues),
        )

    def test_answer_tool_autocorrects_task163_total_value_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            context_dir = task_dir / "context"
            context_dir.mkdir()
            task = PublicTask(
                record=TaskRecord(
                    task_id="task_hidden",
                    difficulty="medium",
                    question=(
                        "Identify the type of expenses and their total value approved "
                        "for 'October Meeting' event."
                    ),
                ),
                assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
            )
            _write_task163_event_expense_context(task.context_dir)

            result = create_default_tool_registry().execute(
                task,
                "answer",
                {"columns": ["type", "total_value"], "rows": [["Meeting", 175.39]]},
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.content["reason"], "answer_validator_autocorrected")
        self.assertEqual(result.answer.columns, ["type", "SUM(T3.cost)"])
        self.assertEqual(result.answer.rows, [["Meeting", 175.39]])

    def test_answer_tool_autocorrects_task27_total_cost_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            context_dir = task_dir / "context"
            context_dir.mkdir()
            task = PublicTask(
                record=TaskRecord(
                    task_id="task_hidden",
                    difficulty="easy",
                    question=(
                        'List out the full name and total cost that member id '
                        '"rec4BLdZHS2Blfp4v" incurred?'
                    ),
                ),
                assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
            )
            _write_member_expense_context(task.context_dir)

            result = create_default_tool_registry().execute(
                task,
                "answer",
                {
                    "columns": ["first_name", "last_name", "total_cost"],
                    "rows": [["Sacha", "Harrison", 866.25]],
                },
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.is_terminal)
        self.assertEqual(result.content["reason"], "answer_validator_autocorrected")
        self.assertEqual(result.answer.columns, ["first_name", "last_name", "SUM(T2.cost)"])
        self.assertEqual(result.answer.rows, [["Sacha", "Harrison", 866.25]])

    def test_member_total_cost_rule_requires_matching_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                'List out the full name and total cost that member id "rec4BLdZHS2Blfp4v" incurred?',
            )

            issues = validate_answer(
                task,
                columns=["first_name", "last_name", "total_cost"],
                rows=[["Sacha", "Harrison", 866.25]],
            )

        self.assertNotIn(
            "member_total_cost_requires_split_name_and_sum_column",
            _issue_codes(issues),
        )

    def test_rejects_total_atoms_counting_all_atoms_in_target_molecules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.",
            )
            _write_csv(
                task.context_dir / "csv" / "atom.csv",
                [
                    {"atom_id": "TR499_1", "molecule_id": "TR499", "element": "p"},
                    {"atom_id": "TR499_2", "molecule_id": "TR499", "element": "c"},
                    {"atom_id": "TR499_3", "molecule_id": "TR499", "element": "c"},
                    {"atom_id": "TR499_4", "molecule_id": "TR499", "element": "h"},
                ],
            )
            _write_sqlite_table(
                task.context_dir / "db" / "bond.db",
                "bond",
                ["bond_id", "molecule_id", "bond_type"],
                [("b1", "TR499", "#")],
            )

            issues = validate_answer(
                task,
                columns=["total_atoms"],
                rows=[[4]],
            )

        self.assertIn(
            "element_atom_count_counts_requested_elements_only",
            _issue_codes(issues),
        )

    def test_accepts_total_atoms_counting_requested_elements_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.",
            )
            _write_csv(
                task.context_dir / "csv" / "atom.csv",
                [
                    {"atom_id": "TR499_1", "molecule_id": "TR499", "element": "p"},
                    {"atom_id": "TR499_2", "molecule_id": "TR499", "element": "c"},
                ],
            )
            _write_sqlite_table(
                task.context_dir / "db" / "bond.db",
                "bond",
                ["bond_id", "molecule_id", "bond_type"],
                [("b1", "TR499", "#")],
            )

            issues = validate_answer(
                task,
                columns=["COUNT(T1.atom_id)"],
                rows=[[1]],
            )

        self.assertNotIn(
            "element_atom_count_counts_requested_elements_only",
            _issue_codes(issues),
        )

    def test_rejects_last_posted_user_alias_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time.",
            )
            _write_json_records(
                task.context_dir / "json" / "posts.json",
                "posts",
                [
                    {
                        "Id": 8222,
                        "Title": "Computer game datasets",
                        "ViewCount": 1708,
                        "OwnerUserId": 37,
                        "LastEditorUserId": 88,
                    }
                ],
            )
            _write_json_records(
                task.context_dir / "json" / "users.json",
                "users",
                [{"Id": 88, "DisplayName": "mbq"}],
            )

            issues = validate_answer(
                task,
                columns=["total_views", "user_name"],
                rows=[[1708, "mbq"]],
            )

        self.assertIn(
            "last_posted_user_requires_last_editor_display_name",
            _issue_codes(issues),
        )

    def test_accepts_last_posted_user_source_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time.",
            )
            _write_json_records(
                task.context_dir / "json" / "posts.json",
                "posts",
                [{"Id": 8222, "ViewCount": 1708, "LastEditorUserId": 88}],
            )
            _write_json_records(
                task.context_dir / "json" / "users.json",
                "users",
                [{"Id": 88, "DisplayName": "mbq"}],
            )

            issues = validate_answer(
                task,
                columns=["ViewCount", "DisplayName"],
                rows=[[1708, "mbq"]],
            )

        self.assertNotIn(
            "last_posted_user_requires_last_editor_display_name",
            _issue_codes(issues),
        )

    def test_rejects_comment_question_returning_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Among the posts with views ranging from 100 to 150, what is the comment with the highest score?",
            )
            _write_csv(
                task.context_dir / "csv" / "comments.csv",
                [{"Id": 90813, "PostId": 10, "Score": 7, "Text": "Full comment"}],
            )

            issues = validate_answer(
                task,
                columns=["Id"],
                rows=[[90813]],
            )

        self.assertIn("comment_question_requires_text_column", _issue_codes(issues))

    def test_accepts_comment_question_returning_full_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Among the posts with views ranging from 100 to 150, what is the comment with the highest score?",
            )
            _write_csv(
                task.context_dir / "csv" / "comments.csv",
                [{"Id": 90813, "PostId": 10, "Score": 7, "Text": "Full comment"}],
            )

            issues = validate_answer(
                task,
                columns=["Text"],
                rows=[["Full comment"]],
            )

        self.assertNotIn("comment_question_requires_text_column", _issue_codes(issues))

    def test_rejects_aggregate_with_unrequested_zero_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "What is the average weight of all female superheroes?")
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT AVG(CAST(weight_kg AS DOUBLE)) AS avg_weight "
                            "FROM superhero WHERE gender_id = 2 AND weight_kg > 0"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["average_weight"],
                rows=[["78.50694444444444"]],
                previous_steps=previous_steps,
            )

        self.assertIn("aggregate_unrequested_zero_exclusion", _issue_codes(issues))

    def test_accepts_aggregate_try_cast_without_zero_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(Path(tmp_dir), "What is the average weight of all female superheroes?")
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT AVG(TRY_CAST(weight_kg AS DOUBLE)) AS avg_weight "
                            "FROM superhero WHERE gender_id = 2"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["average_weight"],
                rows=[["60.77956989247312"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("aggregate_unrequested_zero_exclusion", _issue_codes(issues))

    def test_allows_zero_exclusion_when_question_explicitly_asks_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "What is the average weight of female superheroes with positive weight?",
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            "SELECT AVG(CAST(weight_kg AS DOUBLE)) AS avg_weight "
                            "FROM superhero WHERE gender_id = 2 AND weight_kg > 0"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["average_weight"],
                rows=[["78.50694444444444"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("aggregate_unrequested_zero_exclusion", _issue_codes(issues))

    def test_rejects_raw_price_filter_for_per_unit_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "For all the people who paid more than 29.00 per unit of product id No.5. "
                    "Give their consumption status in the August of 2012."
                ),
            )
            _write_csv(
                task.context_dir / "transactions_1k.csv",
                [{"CustomerID": 5443, "ProductID": 5, "Amount": 2, "Price": 58.04}],
            )
            previous_steps = [
                _step(
                    "execute_context_sql",
                    {
                        "sql": (
                            "SELECT DISTINCT CustomerID FROM transactions_1k "
                            "WHERE ProductID = 5 AND Price > 29.00"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["Consumption"],
                rows=[[88265.39]],
                previous_steps=previous_steps,
            )

        self.assertIn("per_unit_price_requires_amount_division", _issue_codes(issues))

    def test_accepts_unit_price_filter_for_per_unit_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "For all the people who paid more than 29.00 per unit of product id No.5. "
                    "Give their consumption status in the August of 2012."
                ),
            )
            _write_csv(
                task.context_dir / "transactions_1k.csv",
                [{"CustomerID": 5443, "ProductID": 5, "Amount": 2, "Price": 58.04}],
            )
            previous_steps = [
                _step(
                    "execute_context_sql",
                    {
                        "sql": (
                            "SELECT DISTINCT CustomerID FROM transactions_1k "
                            "WHERE ProductID = 5 AND Price * 1.0 / Amount > 29.00"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["Consumption"],
                rows=[[88265.39]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("per_unit_price_requires_amount_division", _issue_codes(issues))

    def test_rejects_customer_id_for_consumption_status_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "For all the people who paid more than 29.00 per unit of product id No.5. "
                    "Give their consumption status in the August of 2012."
                ),
            )

            issues = validate_answer(
                task,
                columns=["CustomerID", "Consumption"],
                rows=[[5443, 88265.39]],
            )

        self.assertIn("consumption_status_extra_customer_id", _issue_codes(issues))

    def test_rejects_empty_gas_station_country_answer_when_month_join_has_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Please list the countries of the gas stations with transactions taken place in June, 2013.",
            )
            _write_csv(
                task.context_dir / "csv" / "yearmonth.csv",
                [
                    {"CustomerID": 44, "Date": 201306, "Consumption": 10.0},
                    {"CustomerID": 45, "Date": 201306, "Consumption": 12.0},
                    {"CustomerID": 999, "Date": 201305, "Consumption": 5.0},
                ],
            )
            _write_json_records(
                task.context_dir / "json" / "gasstations.json",
                "gasstations",
                [
                    {"GasStationID": 44, "Country": "CZE", "Segment": "Value for money"},
                    {"GasStationID": 45, "Country": "SVK", "Segment": "Premium"},
                    {"GasStationID": 999, "Country": "AUT", "Segment": "Other"},
                ],
            )

            issues = validate_answer(
                task,
                columns=["Country"],
                rows=[],
            )

        self.assertIn("empty_gas_station_country_answer", _issue_codes(issues))

    def test_accepts_non_empty_gas_station_country_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Please list the countries of the gas stations with transactions taken place in June, 2013.",
            )
            _write_csv(
                task.context_dir / "csv" / "yearmonth.csv",
                [{"CustomerID": 44, "Date": 201306, "Consumption": 10.0}],
            )
            _write_json_records(
                task.context_dir / "json" / "gasstations.json",
                "gasstations",
                [{"GasStationID": 44, "Country": "CZE", "Segment": "Value for money"}],
            )

            issues = validate_answer(
                task,
                columns=["Country"],
                rows=[["CZE"]],
            )

        self.assertNotIn("empty_gas_station_country_answer", _issue_codes(issues))

    def test_rejects_california_school_answer_without_sat_math_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "List the names and funding types of schools from Riverside-related school "
                    "districts where the average SAT math score across schools exceeds 400."
                ),
            )
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {
                        "sql": (
                            'SELECT "School Name", "Charter Funding Type" FROM frpm '
                            'WHERE "District Name" IN '
                            "('Riverside County Office of Education', 'Riverside Unified')"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["School Name", "Charter Funding Type"],
                rows=[["Riverside County Education Academy", "Locally funded"]],
                previous_steps=previous_steps,
            )

        self.assertIn("california_schools_missing_sat_math_filter", _issue_codes(issues))

    def test_accepts_california_school_answer_with_sat_math_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                (
                    "List the names and funding types of schools from Riverside-related school "
                    "districts where the average SAT math score across schools exceeds 400."
                ),
            )
            previous_steps = [
                _step(
                    "execute_python",
                    {
                        "code": (
                            "merged = frpm.merge(satscores, left_on='CDSCode', right_on='cds')\n"
                            "result = merged[(merged['rtype'] == 'S') & "
                            "(merged['AvgScrMath'] > 400)]"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["sname", "Charter Funding Type"],
                rows=[["Arlington High", ""]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("california_schools_missing_sat_math_filter", _issue_codes(issues))

    def test_rejects_tally_answer_with_count_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
            )

            issues = validate_answer(
                task,
                columns=["element", "count"],
                rows=[["c", 76], ["br", 4]],
            )

        self.assertIn("tally_distinct_projection", _issue_codes(issues))

    def test_rejects_tally_answer_with_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
            )

            issues = validate_answer(
                task,
                columns=["element"],
                rows=[["c"], ["c"], ["br"]],
            )

        self.assertIn("tally_distinct_projection", _issue_codes(issues))

    def test_rejects_toxicology_nth_atom_like_suffix_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
            )
            _write_toxicology_context(task.context_dir)
            previous_steps = [
                _step(
                    "execute_context_duckdb",
                    {"sql": "SELECT DISTINCT element FROM atom WHERE atom_id LIKE '%_4'"},
                )
            ]

            issues = validate_answer(
                task,
                columns=["element"],
                rows=[["br"], ["c"], ["h"], ["na"]],
                previous_steps=previous_steps,
            )

        self.assertIn("toxicology_nth_atom_distinct_elements", _issue_codes(issues))

    def test_rejects_toxicology_nth_atom_lexicographic_rank_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
            )
            _write_toxicology_context(task.context_dir)
            previous_steps = [
                _step(
                    "execute_python",
                    {
                        "code": (
                            "atoms = atoms.sort_values(['molecule_id', 'atom_id'])\n"
                            "atoms['atom_rank'] = atoms.groupby('molecule_id').cumcount() + 1"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["element"],
                rows=[["c"], ["h"]],
                previous_steps=previous_steps,
            )

        self.assertIn("toxicology_nth_atom_distinct_elements", _issue_codes(issues))

    def test_accepts_toxicology_nth_atom_distinct_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _task(
                Path(tmp_dir),
                "Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.",
            )
            _write_toxicology_context(task.context_dir)
            previous_steps = [
                _step(
                    "execute_python",
                    {
                        "code": (
                            "atoms['atom_num'] = atoms['atom_id'].str.split('_').str[-1].astype(int)\n"
                            "result = atoms[atoms['atom_num'] == 4]['element'].drop_duplicates()"
                        )
                    },
                )
            ]

            issues = validate_answer(
                task,
                columns=["element"],
                rows=[["c"], ["br"]],
                previous_steps=previous_steps,
            )

        self.assertNotIn("tally_distinct_projection", _issue_codes(issues))
        self.assertNotIn("toxicology_nth_atom_distinct_elements", _issue_codes(issues))


if __name__ == "__main__":
    unittest.main()
