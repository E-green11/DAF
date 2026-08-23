#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
CLEVR_DATA_ROOT=${CLEVR_DATA_ROOT:-$DATA_ROOT}
CLEVR_SPLIT_MANIFEST="$REPO_ROOT/configs/clevr_spt_split_sha256.txt"
MODEL=${MODEL:-vit_base_patch16_224}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/checkpoints/ViT-B_16.npz}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/reproduce_table1_vit_b16_npz_test_best}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/logs/reproduce_table1_vit_b16_npz_test_best}
GPU_LIST=${GPU_LIST-0}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
NUM_WORKERS=${NUM_WORKERS:-8}
SAVE_EVERY=${SAVE_EVERY:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-0}

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_table1_vit_b16_npz_test_best.sh [--check|--help]

Run the fixed 19-task ViT-B/16 NPZ workflow. Environment overrides include
PYTHON, DATA_ROOT, CLEVR_DATA_ROOT, CHECKPOINT, OUTPUT_ROOT, LOG_ROOT, GPU_LIST,
NUM_WORKERS, BATCH_SIZE, SEED, MODEL, EVAL_INTERVAL, and SAVE_EVERY.

--check  validate Python, GPU_LIST, checkpoint, dataset list files, and the six
         CLEVR SPT/TFDS split-list checksums only
--help   show this message without checking external assets
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
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "error: GPU_LIST must be a comma-separated list of GPU indices: $GPU_LIST" >&2
  exit 1
}
[[ -f "$CHECKPOINT" ]] || {
  echo "error: checkpoint not found: $CHECKPOINT" >&2
  exit 1
}
[[ -d "$DATA_ROOT/cifar-100-python" ]] || {
  echo "error: CIFAR-100 data not found: $DATA_ROOT/cifar-100-python" >&2
  exit 1
}

required_vtab_splits=(
  caltech101:test.txt
  dtd:val200.txt
  oxford_flowers102:val200.txt
  oxford_iiit_pet:val200.txt
  svhn:val200.txt
  sun397:test.txt
  eurosat:val200.txt
  resisc45:val200.txt
  diabetic_retinopathy:val200.txt
  patch_camelyon:val200.txt
  clevr_count:val200.txt
  clevr_dist:val200.txt
  dmlab:val200.txt
  kitti:val200.txt
  dsprites_loc:val200.txt
  dsprites_ori:val200.txt
  smallnorb_azi:val200.txt
  smallnorb_ele:val200.txt
)
missing_data=()
RUN_NAMES=(cifar100)
for spec in "${required_vtab_splits[@]}"; do
  task=${spec%%:*}
  RUN_NAMES+=("$task")
  val_list=${spec#*:}
  task_data_root="$DATA_ROOT"
  if [[ "$task" == "clevr_count" || "$task" == "clevr_dist" ]]; then
    task_data_root="$CLEVR_DATA_ROOT"
  fi
  task_root="$task_data_root/vtab-1k/$task"
  for list_name in train800val200.txt "$val_list" test.txt; do
    [[ -f "$task_root/$list_name" ]] || missing_data+=("$task_root/$list_name")
  done
done
if (( ${#missing_data[@]} > 0 )); then
  echo "error: required VTAB files are missing:" >&2
  printf '  %s\n' "${missing_data[@]}" >&2
  exit 1
fi

[[ -f "$CLEVR_SPLIT_MANIFEST" ]] || {
  echo "error: CLEVR SPT/TFDS split manifest not found: $CLEVR_SPLIT_MANIFEST" >&2
  exit 1
}
if ! "$PYTHON" - "$CLEVR_SPLIT_MANIFEST" "$CLEVR_DATA_ROOT" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
expected_paths = {
    f"vtab-1k/{task}/{list_name}"
    for task in ("clevr_count", "clevr_dist")
    for list_name in ("train800val200.txt", "val200.txt", "test.txt")
}
entries = {}
try:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
except OSError as exc:
    print(f"error: cannot read CLEVR SPT/TFDS split manifest: {exc}", file=sys.stderr)
    raise SystemExit(1)
for line_number, line in enumerate(lines, 1):
    fields = line.split()
    if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        print(
            f"error: invalid CLEVR SPT/TFDS manifest line {line_number}: expected SHA-256 and relative path",
            file=sys.stderr,
        )
        raise SystemExit(1)
    digest, relative = fields
    if relative in entries:
        print(f"error: duplicate CLEVR SPT/TFDS manifest path: {relative}", file=sys.stderr)
        raise SystemExit(1)
    entries[relative] = digest
if set(entries) != expected_paths:
    missing = sorted(expected_paths - set(entries))
    extra = sorted(set(entries) - expected_paths)
    print(
        f"error: CLEVR SPT/TFDS manifest must contain exactly the six split lists; missing={missing}, extra={extra}",
        file=sys.stderr,
    )
    raise SystemExit(1)
for relative, expected_digest in entries.items():
    path = data_root / relative
    try:
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        print(f"error: cannot read CLEVR split list {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if actual_digest != expected_digest:
        print(f"error: CLEVR split-list checksum mismatch: {path}", file=sys.stderr)
        print(f"       expected {expected_digest}", file=sys.stderr)
        print(f"       actual   {actual_digest}", file=sys.stderr)
        print(
            "       use the SPT/TFDS CLEVR split lists required by the reproduction workflow, not an ordinary HF export",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY
then
  exit 1
fi

if (( CHECK_ONLY == 1 )); then
  echo "preflight OK"
  echo "  Python: $PYTHON"
  echo "  checkpoint: $CHECKPOINT"
  echo "  data root: $DATA_ROOT"
  echo "  CLEVR data root: $CLEVR_DATA_ROOT"
  echo "  GPUs: $GPU_LIST"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
LOG_ROOT=$(cd -- "$LOG_ROOT" && pwd -P)
IFS=',' read -ra GPUS <<< "$GPU_LIST"
gpu_count=${#GPUS[@]}
job_index=0
JOB_PIDS=()
JOB_NAMES=()
JOB_OUTPUT_DIRS=()
LOCK_DIRS=()

release_launcher_locks() {
  local index
  for (( index=${#LOCK_DIRS[@]}-1; index>=0; index-- )); do
    rmdir -- "${LOCK_DIRS[$index]}" 2>/dev/null || true
  done
  LOCK_DIRS=()
}

terminate_owned_jobs() {
  local pid
  local job_pid
  local attempt
  local remaining
  local owned_job_pids=()
  for pid in "${JOB_PIDS[@]}"; do
    for job_pid in $(jobs -p); do
      if [[ "$job_pid" == "$pid" ]]; then
        owned_job_pids+=("$pid")
        break
      fi
    done
  done
  JOB_PIDS=()
  for pid in "${owned_job_pids[@]}"; do
    for job_pid in $(jobs -p); do
      if [[ "$job_pid" == "$pid" ]]; then
        kill -TERM "$pid" 2>/dev/null || true
        break
      fi
    done
  done
  for pid in "${owned_job_pids[@]}"; do
    for job_pid in $(jobs -p); do
      if [[ "$job_pid" == "$pid" ]]; then
        kill -CONT "$pid" 2>/dev/null || true
        break
      fi
    done
  done
  for (( attempt=0; attempt<20; attempt++ )); do
    remaining=0
    for pid in "${owned_job_pids[@]}"; do
      for job_pid in $(jobs -p); do
        if [[ "$job_pid" == "$pid" ]]; then
          remaining=1
          break 2
        fi
      done
    done
    (( remaining == 0 )) && break
    sleep 0.05
  done
  for pid in "${owned_job_pids[@]}"; do
    for job_pid in $(jobs -p); do
      if [[ "$job_pid" == "$pid" ]]; then
        kill -KILL "$pid" 2>/dev/null || true
        break
      fi
    done
  done
  for pid in "${owned_job_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  JOB_NAMES=()
  JOB_OUTPUT_DIRS=()
}

cleanup_launcher() {
  local rc=$?
  trap - EXIT INT TERM
  terminate_owned_jobs
  release_launcher_locks
  exit "$rc"
}

trap cleanup_launcher EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

acquire_launcher_lock() {
  local root="$1"
  local lock_dir="$root/.daf-launch.lock"
  if mkdir -- "$lock_dir" 2>/dev/null; then
    LOCK_DIRS+=("$lock_dir")
    return 0
  fi
  echo "error: launcher lock already exists: $lock_dir" >&2
  echo "       another launcher may own this root; choose different roots or wait for it to finish" >&2
  return 1
}

acquire_launcher_lock "$OUTPUT_ROOT"
if [[ "$LOG_ROOT" != "$OUTPUT_ROOT" ]]; then
  acquire_launcher_lock "$LOG_ROOT"
fi

artifact_conflicts=()
for run_name in "${RUN_NAMES[@]}"; do
  output_dir="$OUTPUT_ROOT/$run_name"
  log_file="$LOG_ROOT/$run_name.log"
  if [[ ! -f "$output_dir/.complete" && ( -e "$output_dir" || -e "$log_file" ) ]]; then
    artifact_conflicts+=("$run_name")
  fi
done
if (( ${#artifact_conflicts[@]} > 0 )); then
  echo "error: refusing existing run artifacts without .complete:" >&2
  printf '  %s\n' "${artifact_conflicts[@]}" >&2
  echo "       choose a new OUTPUT_ROOT/LOG_ROOT or explicitly clean the old run artifacts" >&2
  exit 1
fi

wait_for_jobs() {
  local failed=0
  local index
  local rc
  local marker_tmp
  for index in "${!JOB_PIDS[@]}"; do
    if wait "${JOB_PIDS[$index]}"; then
      unset 'JOB_PIDS[index]'
      marker_tmp="${JOB_OUTPUT_DIRS[$index]}/.complete.tmp.$BASHPID.$index"
      if printf 'complete\n' > "$marker_tmp" && mv -- "$marker_tmp" "${JOB_OUTPUT_DIRS[$index]}/.complete"; then
        :
      else
        echo "error: could not create success marker for ${JOB_NAMES[$index]}" >&2
        failed=1
      fi
    else
      rc=$?
      unset 'JOB_PIDS[index]'
      echo "error: training job ${JOB_NAMES[$index]} failed with exit code $rc" >&2
      failed=1
    fi
  done
  JOB_PIDS=()
  JOB_NAMES=()
  JOB_OUTPUT_DIRS=()
  (( failed == 0 ))
}

launch_common() {
  local run_name="$1"
  shift
  local output_dir="$OUTPUT_ROOT/$run_name"
  local log_file="$LOG_ROOT/$run_name.log"

  if [[ -f "$output_dir/.complete" ]]; then
    echo "skip $run_name: success marker exists"
    return 0
  fi
  if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "error: refusing existing run artifacts without .complete for $run_name" >&2
    echo "       choose a new OUTPUT_ROOT/LOG_ROOT or explicitly clean the old run artifacts" >&2
    return 1
  fi

  local gpu="${GPUS[$((job_index % gpu_count))]}"
  echo "launch $run_name on GPU $gpu -> $output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m daf_vit.train \
    --output-dir "$output_dir" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --method daf \
    --image-size 224 \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --save-every "$SAVE_EVERY" \
    --device cuda \
    --amp \
    "$@" > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("$run_name")
  JOB_OUTPUT_DIRS+=("$output_dir")

  job_index=$((job_index + 1))
  if (( job_index % gpu_count == 0 )); then
    if ! wait_for_jobs; then
      return 1
    fi
  fi
}

launch_cifar100() {
  launch_common cifar100 \
    --dataset cifar100 \
    --data-root "$DATA_ROOT" \
    --epochs 200 \
    --train-size 800 \
    --val-size 200 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --warmup-epochs 10 \
    --label-smoothing 0.0 \
    --drop-path-rate 0.0 \
    --rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.0 \
    --budget 0.2 \
    --selection-mode category \
    --update-interval 10 \
    --sensitivity-batches 32 \
    --sensitivity-score-mode grad_square \
    --train-resize-mode random-resized-crop \
    --eval-resize-mode resize-center-crop \
    --norm-mean 0.5,0.5,0.5 \
    --norm-std 0.5,0.5,0.5
}

launch_vtab() {
  local run_name="$1"
  local task="$2"
  local data_root="$3"
  local epochs="$4"
  local lr="$5"
  local label_smoothing="$6"
  local drop_path="$7"
  local rank="$8"
  local alpha="$9"
  local lora_dropout="${10}"
  local selection_mode="${11}"
  local score_mode="${12}"
  local split_qkv="${13}"
  local train_resize="${14}"
  local eval_resize="${15}"
  local norm_mean="${16}"
  local norm_std="${17}"
  local val_list="${18}"

  local extra_args=()
  if [[ "$split_qkv" == "1" ]]; then
    extra_args+=(--split-qkv-lora)
  fi

  launch_common "$run_name" \
    --dataset "vtab:$task" \
    --data-root "$data_root" \
    --epochs "$epochs" \
    --lr "$lr" \
    --weight-decay 0.0001 \
    --warmup-epochs 10 \
    --label-smoothing "$label_smoothing" \
    --drop-path-rate "$drop_path" \
    --rank "$rank" \
    --lora-alpha "$alpha" \
    --lora-dropout "$lora_dropout" \
    --budget 0.2 \
    --selection-mode "$selection_mode" \
    --update-interval 10 \
    --sensitivity-batches 16 \
    --sensitivity-score-mode "$score_mode" \
    --train-resize-mode "$train_resize" \
    --eval-resize-mode "$eval_resize" \
    --vtab-train-list train800val200.txt \
    --vtab-val-list "$val_list" \
    --vtab-test-list test.txt \
    --norm-mean "$norm_mean" \
    --norm-std "$norm_std" \
    "${extra_args[@]}"
}

launch_cifar100

# Natural tasks
launch_vtab caltech101 caltech101 "$DATA_ROOT" 120 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.485,0.456,0.406 0.229,0.224,0.225 test.txt
launch_vtab dtd dtd "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab oxford_flowers102 oxford_flowers102 "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab oxford_iiit_pet oxford_iiit_pet "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab svhn svhn "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab sun397 sun397 "$DATA_ROOT" 120 0.001 0.1 0.0 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 test.txt

# Specialized tasks
launch_vtab eurosat eurosat "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab resisc45 resisc45 "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab diabetic_retinopathy diabetic_retinopathy "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab patch_camelyon patch_camelyon "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt

# Structured tasks
launch_vtab clevr_count clevr_count "$CLEVR_DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab clevr_dist clevr_dist "$CLEVR_DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab dmlab dmlab "$DATA_ROOT" 200 0.001 0.1 0.1 8 16 0.0 category grad_square 0 random-resized-crop resize-center-crop 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab kitti kitti "$DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab dsprites_loc dsprites_loc "$DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab dsprites_ori dsprites_ori "$DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab smallnorb_azi smallnorb_azi "$DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt
launch_vtab smallnorb_ele smallnorb_ele "$DATA_ROOT" 200 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 direct-resize direct-resize 0.5,0.5,0.5 0.5,0.5,0.5 val200.txt

if ! wait_for_jobs; then
  exit 1
fi

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
groups = {
    "Natural": ["cifar100", "caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397"],
    "Specialized": ["eurosat", "resisc45", "diabetic_retinopathy", "patch_camelyon"],
    "Structured": ["clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"],
}
rows = []
for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    metrics = run_dir / "metrics.jsonl"
    args_path = run_dir / "args.json"
    if not metrics.exists() or not args_path.exists():
        continue
    args = json.loads(args_path.read_text())
    dataset = args["dataset"]
    task = "cifar100" if dataset == "cifar100" else dataset.split(":", 1)[1]
    best = None
    for line in metrics.read_text().splitlines():
        row = json.loads(line)
        acc = row.get("test", {}).get("acc1")
        if isinstance(acc, (int, float)) and math.isfinite(acc):
            item = (acc, row.get("epoch"))
            if best is None or item[0] > best[0]:
                best = item
    if best is None:
        continue
    rows.append((task, best[0], best[1]))

rows.sort()
print("| Dataset | Best test | Epoch |")
print("| --- | ---: | ---: |")
values = {}
for task, acc, epoch in rows:
    values[task] = acc
    print(f"| {task} | {acc:.2f} | {epoch} |")

if all(task in values for tasks in groups.values() for task in tasks):
    means = {}
    for name, tasks in groups.items():
        means[name] = sum(values[t] for t in tasks) / len(tasks)
    macro = sum(means.values()) / len(means)
    print()
    for name in ("Natural", "Specialized", "Structured"):
        print(f"{name}: {means[name]:.2f}")
    print(f"Table-1 macro mean: {macro:.2f}")
PY
