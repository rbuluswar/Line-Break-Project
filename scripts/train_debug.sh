#!/usr/bin/env bash
set -euo pipefail

python -m my_project.train \
  --config configs/debug.yaml