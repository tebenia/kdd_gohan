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
