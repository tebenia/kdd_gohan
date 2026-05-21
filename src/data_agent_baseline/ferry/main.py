import os
import shutil
from pathlib import Path

from data_agent_baseline.config import AppConfig, AgentConfig, DatasetConfig, RunConfig
from data_agent_baseline.preprocess import preprocess_input
from data_agent_baseline.run.runner import run_benchmark


def main() -> None:
    model_api_url = os.environ["MODEL_API_URL"]
    model_api_key = os.environ["MODEL_API_KEY"]
    model_name = os.environ.get("MODEL_NAME", "qwen/qwen3.5-35b-a3b")

    input_rw = Path("/tmp/input_rw")
    artifacts_map = preprocess_input(Path("/input"), input_rw)
    if artifacts_map:
        print(f"Preprocessing produced artifacts for {len(artifacts_map)} tasks:", flush=True)
        for tid, descs in artifacts_map.items():
            print(f"  {tid}: {descs}", flush=True)

    config = AppConfig(
        dataset=DatasetConfig(root_path=input_rw),
        agent=AgentConfig(
            model=model_name,
            api_base=model_api_url,
            api_key=model_api_key,
            max_steps=30,
            model_call_timeout_seconds=300,
        ),
        run=RunConfig(
            output_dir=Path("/tmp/runs"),
            run_id="submission",
            max_workers=4,
            task_timeout_seconds=1800,
        ),
    )

    print(f"Starting benchmark: model={model_name}", flush=True)

    _, artifacts = run_benchmark(
        config=config,
        progress_callback=lambda a: print(
            f"[{'OK' if a.succeeded else 'FAIL'}] {a.task_id}", flush=True
        ),
    )

    output_dir = Path("/output")
    copied = 0
    for artifact in artifacts:
        if artifact.prediction_csv_path and artifact.prediction_csv_path.exists():
            task_out = output_dir / artifact.task_id
            task_out.mkdir(parents=True, exist_ok=True)
            shutil.copy(artifact.prediction_csv_path, task_out / "prediction.csv")
            copied += 1

    print(f"Done. {copied}/{len(artifacts)} predictions written to /output", flush=True)


if __name__ == "__main__":
    main()
