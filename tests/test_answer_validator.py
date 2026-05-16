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

    def test_rejects_event_expense_total_value_alias_for_task_163_shape(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
