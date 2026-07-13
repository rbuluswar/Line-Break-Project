#!/usr/bin/env bash
set -euo pipefail

python -m project_main.train \
  --config configs/base.yaml

  