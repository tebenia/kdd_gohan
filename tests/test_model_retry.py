from __future__ import annotations

import unittest

from data_agent_baseline.agents.model import _is_transient_model_error


class ModelRetryTests(unittest.TestCase):
    def test_treats_empty_json_provider_response_as_transient(self) -> None:
        exc = RuntimeError("Expecting value: line 1 column 1 (char 0)")

        self.assertTrue(_is_transient_model_error(exc))

    def test_treats_rate_limit_as_transient(self) -> None:
        exc = RuntimeError("429 limit_requests")

        self.assertTrue(_is_transient_model_error(exc))

    def test_does_not_retry_unrelated_errors(self) -> None:
        exc = RuntimeError("Unknown model name in config")

        self.assertFalse(_is_transient_model_error(exc))


if __name__ == "__main__":
    unittest.main()
