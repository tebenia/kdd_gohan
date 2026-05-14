from __future__ import annotations

import unittest
from unittest.mock import patch

from data_agent_baseline.agents.model import OpenAIModelAdapter, _is_transient_model_error


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

    def test_model_request_timeout_defaults_to_sixty_seconds(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            adapter = OpenAIModelAdapter(
                model="test-model",
                api_base="https://example.test/v1",
                api_key="test-key",
                temperature=0.0,
            )

        self.assertEqual(adapter.request_timeout_seconds, 60.0)

    def test_model_request_timeout_can_be_configured_by_env(self) -> None:
        with patch.dict("os.environ", {"MODEL_REQUEST_TIMEOUT_SECONDS": "12.5"}):
            adapter = OpenAIModelAdapter(
                model="test-model",
                api_base="https://example.test/v1",
                api_key="test-key",
                temperature=0.0,
            )

        self.assertEqual(adapter.request_timeout_seconds, 12.5)


if __name__ == "__main__":
    unittest.main()
