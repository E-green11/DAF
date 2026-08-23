#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_vtab19.sh [--check|--help]

Compatibility notice for the original placeholder launcher. Use
scripts/reproduce_table1_vit_b16_npz_test_best.sh for the runnable workflow.
EOF
}

case "${1:-}" in
  "") ;;
  --check)
    command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: Python executable not found: $PYTHON" >&2; exit 1; }
    echo "preflight OK"
    exit 0
    ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

echo "This repository currently provides a fully runnable torchvision-backed VTAB-CIFAR100 path."
echo "Run $SCRIPT_DIR/train_vtab_cifar100.sh first."
echo "For the full workflow, use $SCRIPT_DIR/reproduce_table1_vit_b16_npz_test_best.sh."
