from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig, parse_model_step
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolRegistry,
)


def _public_task(task_dir: Path) -> PublicTask:
    context_dir = task_dir / "context"
    context_dir.mkdir()
    return PublicTask(
        record=TaskRecord(
            task_id="task_multi_action",
            difficulty="easy",
            question="Test question?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class ReActMultiActionTests(unittest.TestCase):
    def test_parse_model_step_accepts_actions_array(self) -> None:
        step = parse_model_step(
            """```json
{"thought":"parallel","actions":[{"action":"first","action_input":{"x":1}},{"action":"execute_context_duckdb","action_input":"SELECT 1"}]}
```"""
        )

        self.assertEqual(step.thought, "parallel")
        self.assertEqual([action.action for action in step.actions], ["first", "execute_context_duckdb"])
        self.assertEqual(step.actions[1].action_input, {"sql": "SELECT 1"})
        self.assertEqual(step.action, "first")

    def test_parse_model_step_keeps_single_action_fallback(self) -> None:
        step = parse_model_step(
            'prefix {"thought":"single","action":"execute_python","action_input":"print(1)"} suffix'
        )

        self.assertEqual(step.action, "execute_python")
        self.assertEqual(step.action_input, {"code": "print(1)"})

    def test_agent_executes_multiple_actions_in_one_step(self) -> None:
        calls: list[tuple[str, dict[str, object], int]] = []

        def handler(
            _task: PublicTask,
            action_input: dict[str, object],
            context: ToolExecutionContext,
        ) -> ToolExecutionResult:
            name = str(action_input["name"])
            calls.append((name, dict(action_input), len(context.previous_steps)))
            return ToolExecutionResult(ok=True, content={"name": name})

        response = """```json
{"thought":"parallel","actions":[{"action":"tool_a","action_input":{"name":"a"}},{"action":"tool_b","action_input":{"name":"b"}}]}
```"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            task = _public_task(Path(tmp_dir))
            agent = ReActAgent(
                model=ScriptedModelAdapter([response]),
                tools=ToolRegistry(specs={}, handlers={"tool_a": handler, "tool_b": handler}),
                config=ReActAgentConfig(max_steps=1),
            )
            result = agent.run(task)

        self.assertEqual(result.failure_reason, "Agent did not submit an answer within max_steps.")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action, "tool_a,tool_b")
        self.assertEqual(result.steps[0].action_input, [{"name": "a"}, {"name": "b"}])
        self.assertEqual(result.steps[0].observation[0]["tool"], "tool_a")
        self.assertEqual(result.steps[0].observation[1]["tool"], "tool_b")
        self.assertCountEqual([call[0] for call in calls], ["a", "b"])
        self.assertTrue(all(previous_count == 0 for _, _, previous_count in calls))


if __name__ == "__main__":
    unittest.main()
