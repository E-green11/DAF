#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
SPT_ROOT=${SPT_ROOT:-$REPO_ROOT/SPT}
TARGET_ROOT=${TARGET_ROOT:-$DATA_ROOT}
DEFAULT_TASKS="clevr_count clevr_dist"
TASKS=${TASKS:-$DEFAULT_TASKS}

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_vtab1k_spt.sh [--check|--help]

Export the two CLEVR tasks from an existing SPT checkout into
DATA_ROOT/vtab-1k. Override PYTHON, DATA_ROOT, TARGET_ROOT, SPT_ROOT,
or TASKS. The default 2 TASKS are clevr_count and clevr_dist.
EOF
}

CHECK_ONLY=0
case "${1:-}" in
  "") ;;
  --check) CHECK_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "error: Python executable not found: $PYTHON" >&2
  exit 1
}

if [[ ! -f "$SPT_ROOT/data/vtab-source/get_vtab1k.py" ]]; then
  echo "error: SPT get_vtab1k.py not found under $SPT_ROOT" >&2
  exit 1
fi
if (( CHECK_ONLY == 1 )); then
  echo "preflight OK"
  exit 0
fi

mkdir -p "$TARGET_ROOT"
mkdir -p "$TARGET_ROOT/vtab-1k"

if [[ -n "$TASKS" ]]; then
  read -ra task_args <<< "$TASKS"
  "$PYTHON" "$SCRIPT_DIR/export_vtab1k_spt.py" \
    --spt-root "$SPT_ROOT" \
    --output-root "$TARGET_ROOT/vtab-1k" \
    --skip-existing \
    --continue-on-error \
    --tasks "${task_args[@]}"
else
  "$PYTHON" "$SCRIPT_DIR/export_vtab1k_spt.py" \
    --spt-root "$SPT_ROOT" \
    --output-root "$TARGET_ROOT/vtab-1k" \
    --skip-existing \
    --continue-on-error
fi
