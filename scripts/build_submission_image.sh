#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <team_id> <version_number>" >&2
  echo "Example: $0 team0042 3" >&2
  exit 2
fi

TEAM_ID="$1"
VERSION_NUMBER="$2"
IMAGE_NAME="${TEAM_ID}:v${VERSION_NUMBER}"
ARCHIVE_NAME="${TEAM_ID}_v${VERSION_NUMBER}.tar.gz"

docker build --platform=linux/amd64 -t "${IMAGE_NAME}" .
docker save "${IMAGE_NAME}" | gzip > "${ARCHIVE_NAME}"

echo "Built image: ${IMAGE_NAME}"
echo "Wrote archive: ${ARCHIVE_NAME}"
