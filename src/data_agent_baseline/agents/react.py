from __future__ import annotations

import json
import os
import re
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAction, ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolExecutionContext, ToolExecutionResult, ToolRegistry

_SIGALRM_SUPPORTED = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class _StepTimeoutError(TimeoutError):
    pass


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise _StepTimeoutError("Model API call timed out.")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value.strip())


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if json_blocks:
        return json_blocks[-1].strip()
    generic_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    try:
        payload, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON format. Did you forget to escape newlines as \\\\n or double quotes as \\\\\"? "
            f"Error details: {exc}"
        ) from exc
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")

    actions_payload = payload.get("actions")
    if (actions_payload is None or actions_payload == []) and "action" in payload:
        actions_payload = [{"action": payload.get("action"), "action_input": payload.get("action_input", {})}]
    if not isinstance(actions_payload, list):
        raise ValueError("actions must be a list.")
    if not actions_payload:
        raise ValueError("Model response must contain at least one action.")

    actions: list[ModelAction] = []
    for action_item in actions_payload:
        if not isinstance(action_item, dict):
            raise ValueError("Each actions item must be a JSON object.")
        action = action_item.get("action")
        action_input = action_item.get("action_input", {})
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string.")

        if isinstance(action_input, str):
            if action == "execute_python":
                action_input = {"code": action_input}
            elif action in {"execute_context_sql", "execute_context_duckdb"}:
                action_input = {"sql": action_input}

        if not isinstance(action_input, dict):
            raise ValueError("action_input must be a JSON object.")
        actions.append(ModelAction(action=action, action_input=action_input))

    return ModelStep(
        thought=thought,
        actions=actions,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.step_timeout_seconds = max(_env_float("AGENT_STEP_TIMEOUT_SECONDS", 60.0), 0.0)

    @staticmethod
    def _step_actions(step: StepRecord) -> list[str]:
        if not step.action:
            return []
        if step.action == "__error__":
            return ["__error__"]
        return [action.strip() for action in step.action.split(",") if action.strip()]

    @staticmethod
    def _step_action_inputs(step: StepRecord) -> list[dict[str, object]]:
        if isinstance(step.action_input, list):
            return [item for item in step.action_input if isinstance(item, dict)]
        if isinstance(step.action_input, dict):
            return [step.action_input]
        return []

    @staticmethod
    def _step_observations(step: StepRecord) -> list[dict[str, object]]:
        if isinstance(step.observation, list):
            return [item for item in step.observation if isinstance(item, dict)]
        if isinstance(step.observation, dict):
            return [step.observation]
        return []

    @staticmethod
    def _knowledge_text_from_observation(observation: dict[str, object]) -> str | None:
        content = observation.get("content")
        if isinstance(content, dict):
            output = content.get("output")
            if isinstance(output, str) and output.strip():
                return output
            nested_content = content.get("content")
            if isinstance(nested_content, str) and nested_content.strip():
                return nested_content
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, str) and content.strip():
            return content
        return None

    def _find_knowledge_content(self, state: AgentRuntimeState) -> str | None:
        knowledge_content = None
        for step in state.steps:
            actions = self._step_actions(step)
            inputs = self._step_action_inputs(step)
            observations = self._step_observations(step)
            for index, action_input in enumerate(inputs):
                action = actions[index] if index < len(actions) else ""
                observation = observations[index] if index < len(observations) else {}
                if observation.get("ok") is not True:
                    continue
                is_knowledge_read = (
                    action in {"read_doc", "read_json"}
                    and "knowledge.md" in str(action_input.get("path", ""))
                ) or (
                    action == "execute_python"
                    and "knowledge.md" in str(action_input.get("code", ""))
                )
                if is_knowledge_read:
                    knowledge_content = self._knowledge_text_from_observation(observation)
        return knowledge_content

    @staticmethod
    def _build_short_observation(step: StepRecord, observation: dict[str, object]) -> str:
        tool = observation.get("tool") or step.action
        status = "succeeded" if observation.get("ok") else "failed"
        summary = f"Summary: {tool} call {status}."
        error = observation.get("error")
        if error:
            summary += f" Error: {error}"
        return (
            "Observation:\n"
            f"{summary}\n"
            "[Truncated to save context. Use the earlier result if it is relevant, "
            "or query the source again with a narrower query.]"
        )

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))

        if not state.steps:
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "Planning phase: in your first thought, state the independent data "
                        "branches, any required sequence, and how you will merge or verify them. "
                        "Your first action should still read knowledge.md in full when that file "
                        "exists, because it may define schema-specific meanings and examples."
                    ),
                )
            )

        knowledge_content = self._find_knowledge_content(state)
        for index, step in enumerate(state.steps):
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            is_recent = index >= len(state.steps) - 3
            observation_messages: list[str] = []
            for observation in self._step_observations(step):
                rendered = build_observation_prompt(observation)
                if not is_recent and len(rendered) > 500:
                    rendered = self._build_short_observation(step, observation)
                observation_messages.append(rendered)
            messages.append(ModelMessage(role="user", content="\n\n".join(observation_messages)))

        if knowledge_content and len(state.steps) > 3:
            messages.append(
                ModelMessage(
                    role="system",
                    content=f"Reminder: full knowledge.md content previously read:\n{knowledge_content}",
                )
            )

        if len(state.steps) >= 3:
            last_three = state.steps[-3:]
            if all(step.ok for step in last_three):
                action_keys = [step.action for step in last_three]
                input_keys = [json.dumps(step.action_input, ensure_ascii=False, sort_keys=True) for step in last_three]
                if len(set(action_keys)) == 1 and len(set(input_keys)) == 1:
                    messages.append(
                        ModelMessage(
                            role="user",
                            content=(
                                "Success loop detected: you have repeated the exact same "
                                "successful action and input three times. Use the result you "
                                "already have, switch to a different query, or call answer."
                            ),
                        )
                    )

            if all(not step.ok for step in last_three):
                messages.append(
                    ModelMessage(
                        role="user",
                        content=(
                            "Recovery required: the last three steps failed. Change approach. "
                            "If JSON formatting caused the failure, call a simple tool first. "
                            "If Python escaping caused the failure, prefer execute_context_sql "
                            "for databases or read_json/pandas for JSON files. If answer was "
                            "rejected, simplify the final projection to only requested columns."
                        ),
                    )
                )

        steps_remaining = self.config.max_steps - len(state.steps)
        has_answer_attempt = any("answer" in self._step_actions(step) for step in state.steps)
        if state.steps and 1 <= steps_remaining <= 4 and not has_answer_attempt:
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        f"Convergence phase: only {steps_remaining} step(s) remaining. "
                        "Merge gathered data now, verify the row count, verify every question "
                        "filter is still applied after joins/merges, drop helper columns, and "
                        "call answer as soon as the self-check passes."
                    ),
                )
            )
        return messages

    def _complete_with_timeout(self, messages: list[ModelMessage]) -> str:
        if self.step_timeout_seconds <= 0:
            return self.model.complete(messages)

        if _SIGALRM_SUPPORTED and threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, self.step_timeout_seconds)
            try:
                return self.model.complete(messages)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)

        result: list[str] = []
        errors: list[BaseException] = []

        def call_model() -> None:
            try:
                result.append(self.model.complete(messages))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=call_model, daemon=True)
        thread.start()
        thread.join(timeout=self.step_timeout_seconds)
        if thread.is_alive():
            raise _StepTimeoutError("Model API call timed out.")
        if errors:
            raise errors[0]
        if not result:
            raise RuntimeError("Model request ended without returning a response.")
        return result[0]

    @staticmethod
    def _has_consecutive_timeouts(state: AgentRuntimeState, count: int = 3) -> bool:
        if len(state.steps) < count:
            return False
        recent_steps = state.steps[-count:]
        for step in recent_steps:
            if step.action != "__error__":
                return False
            observations = ReActAgent._step_observations(step)
            if not any("timed out" in str(observation.get("error", "")).lower() for observation in observations):
                return False
        return True

    def _execute_model_actions(
        self,
        task: PublicTask,
        actions: list[ModelAction],
        state: AgentRuntimeState,
    ) -> list[tuple[ModelAction, ToolExecutionResult]]:
        context = ToolExecutionContext(previous_steps=tuple(state.steps))

        def run_action(action: ModelAction) -> tuple[ModelAction, ToolExecutionResult]:
            return action, self.tools.execute(task, action.action, action.action_input, context)

        if len(actions) == 1:
            return [run_action(actions[0])]

        worker_count = min(5, len(actions))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(run_action, actions))

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            try:
                raw_response = self._complete_with_timeout(self._build_messages(task, state))
                model_step = parse_model_step(raw_response)
                action_results = self._execute_model_actions(task, model_step.actions, state)
                observations: list[dict[str, object]] = []
                terminal_result = None
                ok_all = True
                for action, tool_result in action_results:
                    observation = {
                        "ok": tool_result.ok,
                        "tool": action.action,
                        "content": tool_result.content,
                    }
                    observations.append(observation)
                    ok_all = ok_all and tool_result.ok
                    if tool_result.is_terminal:
                        terminal_result = tool_result

                action_names = [action.action for action in model_step.actions]
                action_inputs = [action.action_input for action in model_step.actions]
                stored_action = action_names[0] if len(action_names) == 1 else ",".join(action_names)
                stored_action_input = action_inputs[0] if len(action_inputs) == 1 else action_inputs
                stored_observation = observations[0] if len(observations) == 1 else observations

                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=stored_action,
                    action_input=stored_action_input,
                    raw_response=raw_response,
                    observation=stored_observation,
                    ok=ok_all,
                )
                state.steps.append(step_record)
                if terminal_result is not None:
                    state.answer = terminal_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                if self._has_consecutive_timeouts(state):
                    state.failure_reason = "Agent aborted: 3 consecutive model timeouts."
                    break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
