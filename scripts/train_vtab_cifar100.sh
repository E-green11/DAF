#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/vtab_cifar100_daf_seed0}
GPU_LIST=${GPU_LIST-0}
NUM_WORKERS=${NUM_WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-0}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_vtab_cifar100.sh [--check|--help]

Run the original AugReg-backed CIFAR-100 smoke configuration. Override PYTHON,
DATA_ROOT, OUTPUT_DIR, GPU_LIST, NUM_WORKERS, BATCH_SIZE, or SEED.
EOF
}

CHECK_ONLY=0
case "${1:-}" in
  "") ;;
  --check) CHECK_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: Python executable not found: $PYTHON" >&2; exit 1; }
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "error: invalid GPU_LIST: $GPU_LIST" >&2; exit 1; }
[[ -d "$DATA_ROOT/cifar-100-python" ]] || { echo "error: CIFAR-100 data not found: $DATA_ROOT/cifar-100-python" >&2; exit 1; }
if (( CHECK_ONLY == 1 )); then echo "preflight OK"; exit 0; fi

CUDA_VISIBLE_DEVICES="$GPU_LIST" "${PYTHON}" -m daf_vit.train \
  --dataset cifar100 \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model vit_base_patch16_224.augreg_in21k \
  --image-size 224 \
  --epochs 200 \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --warmup-epochs 10 \
  --label-smoothing 0.1 \
  --rank 8 \
  --lora-alpha 16 \
  --budget 0.2 \
  --update-interval 10 \
  --sensitivity-batches 32 \
  --eval-interval 5 \
  --strong-aug \
  --amp
