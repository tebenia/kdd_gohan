from __future__ import annotations

import json
import os
import re
import signal
import threading
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolExecutionContext, ToolRegistry

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
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    parsed_text = text
    try:
        payload, end = json.JSONDecoder().raw_decode(parsed_text)
    except json.JSONDecodeError as exc:
        repaired_text = _repair_truncated_json_object(text, exc)
        if repaired_text is None:
            raise ValueError(
                f"Invalid JSON format. Did you forget to escape newlines as \\\\n or double quotes as \\\\\"? "
                f"Error details: {exc}"
            ) from exc
        parsed_text = repaired_text
        payload, end = json.JSONDecoder().raw_decode(parsed_text)
    remainder = parsed_text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def _find_json_key_value_start(text: str, key: str, *, start: int = 0) -> int | None:
    key_match = re.search(rf'"{re.escape(key)}"\s*:', text[start:])
    if key_match is None:
        return None
    value_start = start + key_match.end()
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    return value_start


def _load_json_string_at(text: str, start: int) -> str | None:
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def _load_json_array_at(text: str, start: int) -> list[object] | None:
    if start >= len(text) or text[start] != "[":
        return None

    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return value

    depth = 0
    in_string = False
    escaped = False
    chars: list[str] = []
    for char in text[start:]:
        if not in_string and char == "}" and depth > 0:
            candidate = "".join(chars) + ("]" * depth)
            try:
                repaired_value = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return repaired_value if isinstance(repaired_value, list) else None

        chars.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed_value = json.loads("".join(chars))
                except json.JSONDecodeError:
                    return None
                return parsed_value if isinstance(parsed_value, list) else None
        elif depth < 0:
            return None

    if depth <= 0:
        return None
    try:
        repaired_value = json.loads("".join(chars) + ("]" * depth))
    except json.JSONDecodeError:
        return None
    return repaired_value if isinstance(repaired_value, list) else None


def _recover_malformed_answer_payload(text: str) -> dict[str, object] | None:
    if re.search(r'"action"\s*:\s*"answer"\s*,', text) is None:
        return None

    action_input_start = _find_json_key_value_start(text, "action_input")
    if action_input_start is None:
        return None

    columns_start = _find_json_key_value_start(text, "columns", start=action_input_start)
    rows_start = _find_json_key_value_start(text, "rows", start=action_input_start)
    if columns_start is None or rows_start is None:
        return None

    columns = _load_json_array_at(text, columns_start)
    rows = _load_json_array_at(text, rows_start)
    if (
        columns is None
        or rows is None
        or not all(isinstance(column, str) for column in columns)
        or not all(isinstance(row, list) for row in rows)
    ):
        return None

    thought_start = _find_json_key_value_start(text, "thought")
    thought = ""
    if thought_start is not None:
        thought = _load_json_string_at(text, thought_start) or ""

    return {
        "thought": thought,
        "action": "answer",
        "action_input": {
            "columns": columns,
            "rows": rows,
        },
    }


def _repair_truncated_json_object(text: str, exc: json.JSONDecodeError) -> str | None:
    stripped_text = text.strip()
    if not stripped_text.startswith("{"):
        return None
    if exc.pos < len(stripped_text) - 2:
        return None

    expected_closers: list[str] = []
    repaired_chars: list[str] = []
    made_repair = False
    in_string = False
    escaped = False
    for char in stripped_text:
        repaired_chars.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            expected_closers.append("}")
        elif char == "[":
            expected_closers.append("]")
        elif char in {"}", "]"}:
            while expected_closers and expected_closers[-1] != char:
                repaired_chars.insert(-1, expected_closers.pop())
                made_repair = True
            if not expected_closers or expected_closers.pop() != char:
                return None

    if in_string:
        return None

    made_repair = made_repair or bool(expected_closers)
    if not made_repair:
        return None
    candidate = "".join(repaired_chars) + "".join(reversed(expected_closers))
    try:
        json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    try:
        payload = _load_single_json_object(normalized)
    except ValueError:
        recovered_payload = _recover_malformed_answer_payload(normalized)
        if recovered_payload is None:
            raise
        payload = recovered_payload

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")

    if isinstance(action_input, str):
        if action == "execute_python":
            action_input = {"code": action_input}
        elif action == "execute_context_sql":
            action_input = {"sql": action_input}

    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
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

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
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
        return all(
            step.action == "__error__"
            and "timed out" in str(step.observation.get("error", "")).lower()
            for step in recent_steps
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            try:
                raw_response = self._complete_with_timeout(self._build_messages(task, state))
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(
                    task,
                    model_step.action,
                    model_step.action_input,
                    ToolExecutionContext(previous_steps=tuple(state.steps)),
                )
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
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
