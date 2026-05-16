from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import queue as queue_module
import re
import signal
import threading
from dataclasses import dataclass

# SIGALRM is the primary per-step timeout: it fires at OS level and interrupts
# blocking socket/SSL reads that Python's thread.join(timeout) cannot interrupt.
_SIGALRM_SUPPORTED = hasattr(signal, "SIGALRM")


class _StepTimeoutError(Exception):
    pass


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise _StepTimeoutError("Model API call timed out (SIGALRM)")

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
    ModelAction,
    OpenAIModelAdapter,
)
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    model_call_timeout_seconds: int = 60


def _complete_model_worker(
    model: ModelAdapter,
    messages: list[ModelMessage],
    queue: multiprocessing.Queue,
) -> None:
    try:
        queue.put({"ok": True, "raw_response": model.complete(messages)})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
        )


def _complete_model_with_process_timeout(
    model: ModelAdapter,
    messages: list[ModelMessage],
    *,
    timeout_seconds: int,
) -> str:
    if timeout_seconds <= 0:
        return model.complete(messages)

    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_complete_model_worker,
        args=(model, messages, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        raise RuntimeError(
            f"Model API call timed out after {timeout_seconds}s "
            "(process-level timeout; subprocess was terminated)."
        )

    try:
        result = queue.get(timeout=1.0)
    except queue_module.Empty:
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            raise RuntimeError(f"Model API call subprocess exited with code {exit_code}.")
        raise RuntimeError("Model API call subprocess exited without returning a response.")

    if result.get("ok"):
        raw_response = result.get("raw_response")
        if not isinstance(raw_response, str):
            raise RuntimeError("Model API call subprocess returned non-string response.")
        return raw_response

    error_type = result.get("error_type", "Error")
    error = result.get("error", "unknown error")
    raise RuntimeError(f"Model API call subprocess failed with {error_type}: {error}")


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    # Try to find a JSON block even if it's not the only thing in the response
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if json_blocks:
        return json_blocks[-1].strip() # Take the last one if multiple exist
    
    generic_blocks = re.findall(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()
        
    # If no fence, look for the first '{' and last '}'
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace+1]
        
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

    actions_payload = payload.get("actions", [])
    
    # Fallback to single action if 'actions' is not present
    if not actions_payload and "action" in payload:
        actions_payload = [{"action": payload["action"], "action_input": payload.get("action_input", {})}]
        
    if not isinstance(actions_payload, list):
        raise ValueError("actions must be a list.")

    if not actions_payload:
        raise ValueError("Model response must contain at least one action.")

    actions = []
    for action_item in actions_payload:
        action = action_item.get("action")
        action_input = action_item.get("action_input", {})
        
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string.")

        if isinstance(action_input, str):
            if action == "execute_python":
                action_input = {"code": action_input}

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

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))

        # DAG Phase 1 — PLAN: keep this active until the first real model step.
        # A model-call timeout records an __error__ step, but no plan was produced.
        if not state.steps or all(step.action == "__error__" for step in state.steps):
            messages.append(ModelMessage(
                role="user",
                content=(
                    "PLANNING PHASE: Before calling any data tool, write your DAG execution plan in your `thought`.\n"
                    "  • BRANCHES: List each independent data source/table you need (queries with no dependency between them).\n"
                    "  • SEQUENCE: Which queries must happen in order? (B depends on result of A)\n"
                    "  • CONVERGENCE: How will you JOIN/merge/aggregate across branches to produce the final answer?\n"
                    "  • RULE CHECKS (do these NOW before reading knowledge.md which may contradict them):\n"
                    "    - If question says 'ranked Nth': plan to use `WHERE rank = N` — NOT positionOrder. Rule 28 overrides knowledge.md.\n"
                    "    - If question asks for 'total value/cost' for an event: plan to use SUM(expense.cost) NOT SUM(budget.amount). Rule 32.\n"
                    "    - If answer includes ID columns: plan to cast them to int (not float). Rule 34.\n"
                    "Start executing your plan after writing it."
                )
            ))
        
        # Knowledge persistence: Always keep the content of knowledge.md if it was ever read.
        knowledge_content = None
        for step in state.steps:
            inputs = step.action_input if isinstance(step.action_input, list) else [step.action_input]
            actions = step.action.split(",") if step.action else []
            obs_list = step.observation if isinstance(step.observation, list) else [step.observation]
            
            for act, inp, obs in zip(actions, inputs, obs_list):
                if not isinstance(inp, dict):
                    continue
                is_knowledge_read = (
                    act in ("read_doc", "read_json")
                    and "knowledge.md" in str(inp.get("path", ""))
                ) or (
                    act == "execute_python"
                    and "knowledge.md" in str(inp.get("code", ""))
                )
                if is_knowledge_read and obs.get("ok"):
                    knowledge_content = obs.get("content")
        
        for i, step in enumerate(state.steps):
            if step.raw_response:
                messages.append(ModelMessage(role="assistant", content=step.raw_response))
            
            is_recent = i >= len(state.steps) - 3
            obs_list = step.observation if isinstance(step.observation, list) else [step.observation]
            
            formatted_obs = []
            for obs_dict in obs_list:
                obs_dict = obs_dict.copy()
                obs_content = build_observation_prompt(obs_dict)
                
                if not is_recent and len(obs_content) > 500:
                    summary = f"Summary: {obs_dict.get('tool')} call {'succeeded' if obs_dict.get('ok') else 'failed'}."
                    if not obs_dict.get("ok"):
                        summary += f" Error: {obs_dict.get('error')}"
                    obs_content = f"Observation:\n{summary}\n[Truncated to save context. Use your memory or recall that this data exists.]"
                formatted_obs.append(obs_content)
                
            messages.append(ModelMessage(role="user", content="\n\n".join(formatted_obs)))
            
        # Inject Knowledge if it was found but might have been truncated
        if knowledge_content and len(state.steps) > 3:
            messages.append(ModelMessage(role="system", content=f"REMINDER: Reference Knowledge.md content for rules and metrics:\n{knowledge_content}"))

        # Success-loop detection: same action+input repeated 3+ times successfully
        if len(state.steps) >= 3:
            last_3_loop = state.steps[-3:]
            if all(s.ok for s in last_3_loop):
                loop_actions = [s.action for s in last_3_loop]
                loop_inputs = [str(s.action_input) for s in last_3_loop]
                if len(set(loop_actions)) == 1 and len(set(loop_inputs)) == 1:
                    messages.append(ModelMessage(
                        role="user",
                        content=(
                            "SUCCESS LOOP DETECTED: You have called the exact same action with the same input 3 times in a row. "
                            "Stop repeating it — you already have the result. "
                            "Review your observation history: the data you need is likely already there from an earlier step. "
                            "You MUST switch to a completely different action to make progress."
                        )
                    ))

        # Error recovery: if the last 3 steps are all failures (any type), inject a recovery hint
        if len(state.steps) >= 3:
            last_3 = state.steps[-3:]
            if all(not s.ok for s in last_3):
                messages.append(ModelMessage(
                    role="user",
                    content=(
                        "RECOVERY REQUIRED: You have failed 3 steps in a row. "
                        "You MUST completely change your approach. "
                        "If you have been failing at JSON formatting, output the simplest possible action with NO Python code inside: "
                        '```json\n{"thought":"Restarting with list_context","actions":[{"action":"list_context","action_input":{"max_depth":2}}]}\n``` '
                        "If JSON escaping is the issue, use execute_context_sql instead of execute_python for database queries. "
                        "For JSON files, use the read_json tool (not duckdb.read_json_auto inside execute_python). "
                        "If you have a valid result but keep failing to submit it, simplify your answer columns to only what the question explicitly asks for."
                    )
                ))

        # DAG Phase 3 — CONVERGE & ITERATE: inject convergence/self-check prompt when steps are running low
        steps_remaining = self.config.max_steps - len(state.steps)
        has_answer_attempt = any(s.action == "answer" for s in state.steps)
        if state.steps and 1 <= steps_remaining <= 4 and not has_answer_attempt:
            messages.append(ModelMessage(
                role="user",
                content=(
                    f"CONVERGENCE PHASE — only {steps_remaining} step(s) remaining.\n"
                    "1. MERGE: combine all data gathered from your branches (JOIN, filter, aggregate).\n"
                    "2. SELF-CHECK in your thought:\n"
                    "   • Columns: count them. Return ONLY what the question explicitly asks for. "
                    "If you have Id/Score/date/amount but the question asks for name/text/trans_id only — drop the extras now.\n"
                    "   • Row count = what the question scope implies? (single result → 1 row; "
                    "list with condition → all matching rows, NOT 0 rows unless truly none exist)\n"
                    "   • Filters: ALL WHERE conditions applied? If you merged SQL results with CSV in pandas, "
                    "did you re-apply the SQL filter conditions in pandas too?\n"
                    "3. ITERATE if wrong: identify which branch returned bad data, fix that branch only, re-merge.\n"
                    "4. Call `answer` as soon as self-check passes."
                )
            ))

        return messages

    def _complete_model_with_signal_timeout(self, messages: list[ModelMessage]) -> str:
        # Fallback path for scripted/custom models. Real OpenAIModelAdapter calls use
        # the process timeout below so a wedged TLS/socket call can be killed.
        if self.config.model_call_timeout_seconds <= 0:
            return self.model.complete(messages)

        if _SIGALRM_SUPPORTED:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.config.model_call_timeout_seconds)
            try:
                return self.model.complete(messages)
            finally:
                signal.alarm(0)

        _result: list[str] = []
        _error: list[BaseException] = []

        def _call() -> None:
            try:
                _result.append(self.model.complete(messages))
            except BaseException as _e:
                _error.append(_e)

        _t = threading.Thread(target=_call, daemon=True)
        _t.start()
        _t.join(timeout=self.config.model_call_timeout_seconds)
        if _t.is_alive():
            raise RuntimeError(
                f"Model API call timed out after {self.config.model_call_timeout_seconds}s "
                "(thread-level timeout)."
            )
        if _error:
            raise _error[0]
        if not _result:
            raise RuntimeError("Model API call finished without returning a response.")
        return _result[0]

    def _complete_model(self, messages: list[ModelMessage]) -> str:
        # The benchmark already runs each task in a subprocess, but a hung first
        # model request used to block that whole task until task_timeout_seconds.
        # Running the API call in its own subprocess turns that into a retryable
        # per-step error instead.
        if isinstance(self.model, OpenAIModelAdapter):
            return _complete_model_with_process_timeout(
                self.model,
                messages,
                timeout_seconds=self.config.model_call_timeout_seconds,
            )
        return self._complete_model_with_signal_timeout(messages)

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            try:
                raw_response = self._complete_model(self._build_messages(task, state))
                model_step = parse_model_step(raw_response)
                
                observations = []
                ok_all = True
                is_terminal = False
                
                def run_action(act: ModelAction):
                    return act, self.tools.execute(task, act.action, act.action_input)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(run_action, model_step.actions))
                
                for act, tool_result in results:
                    obs = {
                        "ok": tool_result.ok,
                        "tool": act.action,
                        "content": tool_result.content,
                    }
                    observations.append(obs)
                    if not tool_result.ok:
                        ok_all = False
                    if tool_result.is_terminal:
                        is_terminal = True
                        state.answer = tool_result.answer
                        
                actions_str = ",".join([act.action for act in model_step.actions])
                inputs_list = [act.action_input for act in model_step.actions]
                
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=actions_str,
                    action_input=inputs_list,
                    raw_response=raw_response,
                    observation=observations,
                    ok=ok_all,
                )
                state.steps.append(step_record)
                if is_terminal:
                    break
            except Exception as exc:
                observation = [{
                    "ok": False,
                    "error": str(exc),
                }]
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input=[],
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

            # Early exit: if last 5 steps all timed out, stop wasting the task budget.
            if len(state.steps) >= 5:
                last_5 = state.steps[-5:]
                if all(
                    not s.ok and isinstance(s.observation, list) and len(s.observation) > 0 and "timed out" in s.observation[0].get("error", "").lower()
                    for s in last_5
                ):
                    state.failure_reason = "Agent aborted: 5 consecutive API timeouts."
                    break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
