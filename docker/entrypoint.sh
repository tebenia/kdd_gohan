#!/bin/sh
set -eu

mkdir -p /output /logs

status_file="/tmp/data_agent_submission_status"
rm -f "${status_file}"

{
  python -m data_agent_baseline.submission
  echo "$?" > "${status_file}"
} 2>&1 | tee /logs/runtime.log

exit "$(cat "${status_file}")"
