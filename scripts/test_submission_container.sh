#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image_name>" >&2
  echo "Example: $0 team0042:v3" >&2
  exit 2
fi

IMAGE_NAME="$1"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TEST_ROOT="${PROJECT_ROOT}/tmp_submission_test"

rm -rf "${TEST_ROOT}"
mkdir -p "${TEST_ROOT}/input" "${TEST_ROOT}/output" "${TEST_ROOT}/logs"
cp -R "${PROJECT_ROOT}/data/public/input/." "${TEST_ROOT}/input/"

docker run --rm \
  --platform=linux/amd64 \
  -v "${TEST_ROOT}/input:/input:ro" \
  -v "${TEST_ROOT}/output:/output:rw" \
  -v "${TEST_ROOT}/logs:/logs:rw" \
  -e MODEL_API_URL="${MODEL_API_URL:?MODEL_API_URL is required}" \
  -e MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}" \
  -e MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b}" \
  -e SUBMISSION_MAX_WORKERS="${SUBMISSION_MAX_WORKERS:-1}" \
  -e AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-16}" \
  -e SUBMISSION_TASK_TIMEOUT_SECONDS="${SUBMISSION_TASK_TIMEOUT_SECONDS:-600}" \
  -e SUBMISSION_WRITE_TRACES="${SUBMISSION_WRITE_TRACES:-0}" \
  "${IMAGE_NAME}"

echo "Test output: ${TEST_ROOT}/output"
echo "Test logs:   ${TEST_ROOT}/logs"
