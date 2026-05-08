from __future__ import annotations

import csv
import json
import multiprocessing
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry

DEFAULT_INPUT_ROOT = Path("/input")
DEFAULT_OUTPUT_ROOT = Path("/output")
DEFAULT_LOG_ROOT = Path("/logs")

STOP_REQUESTED = False


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    input_root: Path
    output_root: Path
    log_root: Path
    model_name: str
    model_api_url: str
    model_api_key: str
    max_steps: int
    max_workers: int
    task_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class SubmissionTaskResult:
    task_id: str
    succeeded: bool
    prediction_path: str | None
    elapsed_seconds: float
    failure_reason: str | None = None


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value or not value.strip():
        return default
    return Path(value.strip())


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value or not value.strip():
        return default
    return int(value.strip())


def load_submission_config() -> SubmissionConfig:
    model_api_url = os.environ.get("MODEL_API_URL", "").strip()
    if not model_api_url:
        raise RuntimeError("Missing required environment variable MODEL_API_URL.")

    return SubmissionConfig(
        input_root=_env_path("SUBMISSION_INPUT_ROOT", DEFAULT_INPUT_ROOT),
        output_root=_env_path("SUBMISSION_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        log_root=_env_path("SUBMISSION_LOG_ROOT", DEFAULT_LOG_ROOT),
        model_name=os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b").strip() or "qwen3.5-35b-a3b",
        model_api_url=model_api_url,
        model_api_key=os.environ.get("MODEL_API_KEY", "EMPTY").strip() or "EMPTY",
        max_steps=_env_int("AGENT_MAX_STEPS", 16),
        max_workers=max(_env_int("SUBMISSION_MAX_WORKERS", 1), 1),
        task_timeout_seconds=_env_int("SUBMISSION_TASK_TIMEOUT_SECONDS", 600),
    )


def _handle_stop_signal(signum: int, _: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; finishing in-flight work and stopping new tasks.", flush=True)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)


def task_number(task_id: str) -> int:
    if not task_id.startswith("task_"):
        return 10**9
    try:
        return int(task_id.removeprefix("task_"))
    except ValueError:
        return 10**9


def load_public_task(task_dir: Path) -> PublicTask:
    task_json_path = task_dir / "task.json"
    payload = json.loads(task_json_path.read_text())
    record = TaskRecord(
        task_id=str(payload["task_id"]),
        difficulty=str(payload["difficulty"]),
        question=str(payload["question"]),
    )
    if record.task_id != task_dir.name:
        raise ValueError(f"task_id mismatch for {task_dir}: task.json has {record.task_id}")

    context_dir = task_dir / "context"
    if not context_dir.is_dir():
        raise FileNotFoundError(f"Missing context dir: {context_dir}")
    return PublicTask(record=record, assets=TaskAssets(task_dir=task_dir, context_dir=context_dir))


def iter_task_dirs(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")
    task_dirs = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and path.name.startswith("task_") and (path / "task.json").is_file()
    ]
    task_dirs.sort(key=lambda path: (task_number(path.name), path.name))
    return task_dirs


def build_agent(config: SubmissionConfig) -> ReActAgent:
    model = OpenAIModelAdapter(
        model=config.model_name,
        api_base=config.model_api_url,
        api_key=config.model_api_key,
        temperature=0.0,
    )
    return ReActAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=config.max_steps),
    )


def write_prediction(output_root: Path, task_id: str, answer: AnswerTable) -> Path:
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "prediction.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(answer.columns)
        writer.writerows(answer.rows)
    return prediction_path


def write_empty_prediction(output_root: Path, task_id: str) -> Path:
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = out_dir / "prediction.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer"])
    return prediction_path


def process_task_core(task_dir: Path, config: SubmissionConfig) -> SubmissionTaskResult:
    started_at = perf_counter()
    task_id = task_dir.name
    try:
        task = load_public_task(task_dir)
        agent = build_agent(config)
        result = agent.run(task)
        if result.answer is None:
            prediction_path = write_empty_prediction(config.output_root, task.task_id)
            failure_reason = result.failure_reason or "Agent did not submit an answer."
            return SubmissionTaskResult(
                task_id=task.task_id,
                succeeded=False,
                prediction_path=str(prediction_path),
                elapsed_seconds=round(perf_counter() - started_at, 3),
                failure_reason=failure_reason,
            )

        prediction_path = write_prediction(config.output_root, task.task_id, result.answer)
        return SubmissionTaskResult(
            task_id=task.task_id,
            succeeded=True,
            prediction_path=str(prediction_path),
            elapsed_seconds=round(perf_counter() - started_at, 3),
        )
    except BaseException as exc:  # noqa: BLE001
        prediction_path = write_empty_prediction(config.output_root, task_id)
        return SubmissionTaskResult(
            task_id=task_id,
            succeeded=False,
            prediction_path=str(prediction_path),
            elapsed_seconds=round(perf_counter() - started_at, 3),
            failure_reason=str(exc),
        )


def process_task_in_subprocess(
    task_dir: str,
    config: SubmissionConfig,
    queue: multiprocessing.Queue[Any],
) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    result = process_task_core(Path(task_dir), config)
    queue.put(asdict(result))


def terminate_task_process(process: multiprocessing.Process) -> None:
    if process.pid and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            process.terminate()
    else:
        process.terminate()

    process.join(timeout=1.0)
    if process.is_alive():
        if process.pid and hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                process.kill()
        else:
            process.kill()
        process.join()


def process_task(task_dir: Path, config: SubmissionConfig) -> SubmissionTaskResult:
    timeout_seconds = config.task_timeout_seconds
    if timeout_seconds <= 0:
        return process_task_core(task_dir, config)

    started_at = perf_counter()
    task_id = task_dir.name
    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=process_task_in_subprocess,
        args=(task_dir.as_posix(), config, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        terminate_task_process(process)
        prediction_path = write_empty_prediction(config.output_root, task_id)
        return SubmissionTaskResult(
            task_id=task_id,
            succeeded=False,
            prediction_path=str(prediction_path),
            elapsed_seconds=round(perf_counter() - started_at, 3),
            failure_reason=f"Task timed out after {timeout_seconds} seconds.",
        )

    try:
        payload = queue.get(timeout=1.0)
    except Empty:
        prediction_path = write_empty_prediction(config.output_root, task_id)
        return SubmissionTaskResult(
            task_id=task_id,
            succeeded=False,
            prediction_path=str(prediction_path),
            elapsed_seconds=round(perf_counter() - started_at, 3),
            failure_reason=f"Task exited without returning a result (exit_code={process.exitcode}).",
        )

    return SubmissionTaskResult(**payload)


def write_summary(log_root: Path, payload: dict[str, Any]) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def run_submission(config: SubmissionConfig) -> int:
    started_at = perf_counter()
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] "
        f"Starting submission run input={config.input_root} output={config.output_root} "
        f"model={config.model_name} max_steps={config.max_steps} "
        f"max_workers={config.max_workers} task_timeout_seconds={config.task_timeout_seconds}",
        flush=True,
    )
    task_dirs = iter_task_dirs(config.input_root)
    print(f"Discovered {len(task_dirs)} tasks.", flush=True)

    results: list[SubmissionTaskResult] = []
    if config.max_workers == 1:
        for task_dir in task_dirs:
            if STOP_REQUESTED:
                break
            result = process_task(task_dir, config)
            results.append(result)
            status = "ok" if result.succeeded else "fail"
            print(
                f"{result.task_id}: {status} elapsed={result.elapsed_seconds}s "
                f"prediction={result.prediction_path}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for task_dir in task_dirs:
                if STOP_REQUESTED:
                    break
                futures.append(executor.submit(process_task, task_dir, config))
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "ok" if result.succeeded else "fail"
                print(
                    f"{result.task_id}: {status} elapsed={result.elapsed_seconds}s "
                    f"prediction={result.prediction_path}",
                    flush=True,
                )

    results.sort(key=lambda item: (task_number(item.task_id), item.task_id))
    summary = {
        "task_count": len(task_dirs),
        "completed_task_count": len(results),
        "succeeded_task_count": sum(1 for item in results if item.succeeded),
        "elapsed_seconds": round(perf_counter() - started_at, 3),
        "stopped_early": STOP_REQUESTED,
        "tasks": [asdict(item) for item in results],
    }
    write_summary(config.log_root, summary)
    print(
        f"Finished submission run: {summary['succeeded_task_count']}/{summary['task_count']} "
        f"succeeded in {summary['elapsed_seconds']}s.",
        flush=True,
    )
    return 0


def main() -> int:
    install_signal_handlers()
    config = load_submission_config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.log_root.mkdir(parents=True, exist_ok=True)
    return run_submission(config)


if __name__ == "__main__":
    raise SystemExit(main())
