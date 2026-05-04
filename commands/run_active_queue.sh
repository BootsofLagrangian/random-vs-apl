#!/bin/bash
set -euo pipefail

# Convenience wrapper to launch the Active DAP experiment queue.
# Usage:
#   bash commands/run_active_queue.sh [additional args...]
#
# By default this script points the runner at the public paper Sweep A
# config. Pass CONFIG_PATH or extra flags to override, e.g.:
#   bash commands/run_active_queue.sh --dry-run
#   bash commands/run_active_queue.sh --sanity-check
#   bash commands/run_active_queue.sh --max-concurrency 2 --gpus 0,1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/paper_sweep_A_harmful_multiseed.yaml}"

"${PYTHON:-python3}" "${REPO_ROOT}/scripts/active_queue_runner.py" \
  --config "${CONFIG_PATH}" \
  "$@"
