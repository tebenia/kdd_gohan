# Docker Submission Guide

This project includes a submission entrypoint for the KDD Cup 2026 Data Agents rules.

## Runtime Contract

The container reads:

```text
/input/task_<id>/task.json
/input/task_<id>/context/...
```

and writes:

```text
/output/task_<id>/prediction.csv
/logs/runtime.log
/logs/summary.json
```

Model configuration is read only from runtime environment variables:

```text
MODEL_API_URL
MODEL_API_KEY
MODEL_NAME
SUBMISSION_MAX_WORKERS
SUBMISSION_TASK_TIMEOUT_SECONDS
```

Local-only data, artifacts, notebooks, virtual environments, and local config files are excluded by `.dockerignore`.

## Build

Use the team id assigned by the organizers and increment the version number for every submission:

```bash
./scripts/build_submission_image.sh <team_id> <version_number>
```

Example:

```bash
./scripts/build_submission_image.sh team0042 3
```

This creates:

```text
Image:   team0042:v3
Archive: team0042_v3.tar.gz
```

The archive is created with `docker save`, as required by the rules.

## Local Smoke Test

Set your local model endpoint first:

```bash
export MODEL_API_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_API_KEY="your-api-key"
export MODEL_NAME="qwen3.5-35b-a3b"
export SUBMISSION_MAX_WORKERS=8
export SUBMISSION_TASK_TIMEOUT_SECONDS=600
```

Run:

```bash
./scripts/test_submission_container.sh team0042:v3
```

The script mounts local public demo input into `/input`, writes predictions under `tmp_submission_test/output`, and logs under `tmp_submission_test/logs`.

Each task runs in a child process. If a task exceeds `SUBMISSION_TASK_TIMEOUT_SECONDS`, the process is stopped and an empty `prediction.csv` is written for that task so the remaining tasks can continue.

## Email Submission

Upload `<team_id>_v<N>.tar.gz` to Google Drive and set sharing to "Anyone with the link" as Viewer.

Email the organizers:

```text
Subject: [KDDCup2026 Data Agents] Submission - <team_id> - v<N>

Team ID: <team_id>
Version: v<N>
Sharing link: <google_drive_link>
```

The sender must be the registered team leader email.
