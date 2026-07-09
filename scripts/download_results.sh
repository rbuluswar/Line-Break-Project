#!/usr/bin/env bash
set -euo pipefail

INSTANCE_IP="$1"
KEY_FILE="$2"
REMOTE_DIR="$3"
LOCAL_DIR="$4"

rsync -avz -e "ssh -i ${KEY_FILE}" \
  "ubuntu@${INSTANCE_IP}:${REMOTE_DIR}/" \
  "${LOCAL_DIR}/"