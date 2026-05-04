#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash commands/run_reproduce.sh --sweep A|B [runner args...]

Examples:
  bash commands/run_reproduce.sh --sweep A --dry-run
  bash commands/run_reproduce.sh --sweep B --dry-run
  bash commands/run_reproduce.sh --sweep A --max-concurrency 2 --gpus 0,1
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)
      SWEEP="${2:-}"
      shift 2
      ;;
    --sweep=*)
      SWEEP="${1#--sweep=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "$SWEEP" in
  A|a)
    CONFIG_PATH="${REPO_ROOT}/configs/paper_sweep_A_harmful_multiseed.yaml"
    ;;
  B|b)
    CONFIG_PATH="${REPO_ROOT}/configs/paper_sweep_B_controls.yaml"
    ;;
  *)
    usage
    exit 2
    ;;
esac

"${PYTHON:-python3}" "${REPO_ROOT}/scripts/active_queue_runner.py" --config "${CONFIG_PATH}" "${ARGS[@]}"
