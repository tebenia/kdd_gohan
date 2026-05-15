from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_agent_baseline.config import AppConfig, RunConfig, load_app_config
from data_agent_baseline.run.runner import _worker_count_for_difficulty
from data_agent_baseline.submission import (
    SubmissionConfig,
    load_submission_config,
    task_worker_count,
)


def _write_task_json(task_dir: Path, *, task_id: str, difficulty: str) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": "Test question?",
            }
        )
    )


class WorkerConfigTests(unittest.TestCase):
    def test_local_config_defaults_hard_tasks_to_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                """
run:
  output_dir: artifacts/runs
  max_workers: 4
"""
            )

            config = load_app_config(config_path)

        self.assertEqual(config.run.max_workers, 4)
        self.assertEqual(config.run.difficulty_max_workers, {"hard": 1})

    def test_local_config_uses_difficulty_worker_overrides(self) -> None:
        config = AppConfig(
            run=RunConfig(
                max_workers=4,
                difficulty_max_workers={"hard": 1, "medium": 2},
            )
        )

        self.assertEqual(_worker_count_for_difficulty("easy", config=config), 4)
        self.assertEqual(_worker_count_for_difficulty("medium", config=config), 2)
        self.assertEqual(_worker_count_for_difficulty("hard", config=config), 1)

    def test_submission_config_defaults_to_four_workers_and_one_hard_worker(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_API_URL": "https://example.test/v1",
            },
            clear=True,
        ):
            config = load_submission_config()

        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.difficulty_max_workers, {"hard": 1})

    def test_submission_config_env_overrides_difficulty_workers(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODEL_API_URL": "https://example.test/v1",
                "SUBMISSION_MAX_WORKERS": "6",
                "SUBMISSION_EASY_MAX_WORKERS": "5",
                "SUBMISSION_MEDIUM_MAX_WORKERS": "4",
                "SUBMISSION_HARD_MAX_WORKERS": "2",
            },
            clear=True,
        ):
            config = load_submission_config()

        self.assertEqual(config.max_workers, 6)
        self.assertEqual(
            config.difficulty_max_workers,
            {"easy": 5, "medium": 4, "hard": 2},
        )

    def test_submission_task_worker_count_uses_task_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            easy_dir = root / "task_1"
            hard_dir = root / "task_2"
            _write_task_json(easy_dir, task_id="task_1", difficulty="easy")
            _write_task_json(hard_dir, task_id="task_2", difficulty="hard")
            config = SubmissionConfig(
                input_root=root,
                output_root=root / "output",
                log_root=root / "logs",
                model_name="test-model",
                model_api_url="https://example.test/v1",
                model_api_key="test-key",
                max_steps=1,
                max_workers=4,
                difficulty_max_workers={"hard": 1},
                task_timeout_seconds=1,
            )

            self.assertEqual(task_worker_count(easy_dir, config), 4)
            self.assertEqual(task_worker_count(hard_dir, config), 1)


if __name__ == "__main__":
    unittest.main()
