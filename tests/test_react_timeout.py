from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import ToolRegistry


class SlowModel:
    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        time.sleep(5)
        return ""


class ReActTimeoutTests(unittest.TestCase):
    def test_model_step_timeout_interrupts_slow_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_dir = Path(tmp_dir)
            context_dir = task_dir / "context"
            context_dir.mkdir()
            task = PublicTask(
                record=TaskRecord(
                    task_id="task_timeout",
                    difficulty="easy",
                    question="What is the answer?",
                ),
                assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
            )

            with patch.dict("os.environ", {"AGENT_STEP_TIMEOUT_SECONDS": "0.05"}):
                agent = ReActAgent(
                    model=SlowModel(),
                    tools=ToolRegistry(specs={}, handlers={}),
                    config=ReActAgentConfig(max_steps=1),
                )
                started_at = time.perf_counter()
                result = agent.run(task)
                elapsed_seconds = time.perf_counter() - started_at

        self.assertLess(elapsed_seconds, 1.0)
        self.assertEqual(result.failure_reason, "Agent did not submit an answer within max_steps.")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action, "__error__")
        self.assertIn("timed out", result.steps[0].observation["error"])


if __name__ == "__main__":
    unittest.main()
