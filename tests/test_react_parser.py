from __future__ import annotations

import unittest

from data_agent_baseline.agents.react import parse_model_step


class ReActParserTests(unittest.TestCase):
    def test_repairs_model_response_missing_final_closer(self) -> None:
        raw_response = (
            "```json\n"
            '{"thought":"done","action":"answer","action_input":'
            '{"columns":["first_name","last_name","total_cost"],'
            '"rows":[["Sacha","Harrison",866.25]}}\n'
            "```"
        )

        step = parse_model_step(raw_response)

        self.assertEqual(step.action, "answer")
        self.assertEqual(
            step.action_input,
            {
                "columns": ["first_name", "last_name", "total_cost"],
                "rows": [["Sacha", "Harrison", 866.25]],
            },
        )

    def test_does_not_repair_json_error_away_from_end(self) -> None:
        raw_response = (
            "```json\n"
            '{"thought":"done","action":"answer" "action_input":{"columns":["x"],"rows":[[1]]}}\n'
            "```"
        )

        with self.assertRaisesRegex(ValueError, "Invalid JSON format"):
            parse_model_step(raw_response)

    def test_recovers_malformed_answer_rows_array(self) -> None:
        raw_response = (
            "```json\n"
            "{\"thought\":\"I have the member information (Sacha Harrison) and calculated "
            "the total cost (866.25) for member id 'rec4BLdZHS2Blfp4v'.\","
            "\"action\":\"answer\","
            "\"action_input\":{\"columns\":[\"first_name\",\"last_name\",\"total_cost\"],"
            "\"rows\":[[\"Sacha\",\"Harrison\",866.25]}}}\n"
            "```"
        )

        step = parse_model_step(raw_response)

        self.assertEqual(step.action, "answer")
        self.assertEqual(
            step.action_input,
            {
                "columns": ["first_name", "last_name", "total_cost"],
                "rows": [["Sacha", "Harrison", 866.25]],
            },
        )


if __name__ == "__main__":
    unittest.main()
