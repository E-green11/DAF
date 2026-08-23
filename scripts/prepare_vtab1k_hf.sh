#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$DATA_ROOT/vtab-1k}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$DATA_ROOT/downloads}
DEFAULT_TASKS="caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 eurosat resisc45 diabetic_retinopathy patch_camelyon dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele"
TASKS=${TASKS:-$DEFAULT_TASKS}
ZIP_PATH=${ZIP_PATH:-}

usage() {
  cat <<'EOF'
Usage: bash scripts/prepare_vtab1k_hf.sh [--check|--help]

Prepare VTAB-1k under DATA_ROOT/vtab-1k with the Hugging Face exporter.
The default TASKS are the 16 non-CLEVR VTAB tasks used from this source.
Override PYTHON, DATA_ROOT, OUTPUT_ROOT, DOWNLOAD_DIR, TASKS, or ZIP_PATH.
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
if [[ -n "$ZIP_PATH" && ! -f "$ZIP_PATH" ]]; then
  echo "error: ZIP_PATH does not exist: $ZIP_PATH" >&2
  exit 1
fi
if (( CHECK_ONLY == 1 )); then
  echo "preflight OK"
  exit 0
fi

args=(
  "$SCRIPT_DIR/export_vtab1k_hf.py"
  --output-root "$OUTPUT_ROOT"
  --download-dir "$DOWNLOAD_DIR"
  --skip-existing
)

if [[ -n "$ZIP_PATH" ]]; then
  args+=(--zip-path "$ZIP_PATH")
fi

if [[ -n "$TASKS" ]]; then
  read -ra task_args <<< "$TASKS"
  args+=(--tasks "${task_args[@]}")
fi

"$PYTHON" "${args[@]}"
