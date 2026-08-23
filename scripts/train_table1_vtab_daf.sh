#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/checkpoints/ViT-B_16.npz}
MODEL=${MODEL:-vit_base_patch16_224}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/table1_vtab_vit_b16_npz_tableproto}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/logs/table1_vtab_vit_b16_npz_tableproto}
GPU_LIST=${GPU_LIST-0}

usage() {
  cat <<'EOF'
Usage: bash scripts/train_table1_vtab_daf.sh [--check|--help]

Run a configurable VTAB-1k task set. Override PYTHON, DATA_ROOT, CHECKPOINT,
OUTPUT_ROOT, LOG_ROOT, GPU_LIST, TASKS, and the training variables in this file.
TASKS must be set to a non-empty, space-separated list of VTAB task names.
EOF
}

CHECK_ONLY=0
case "${1:-}" in
  "") ;;
  --check) CHECK_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

EPOCHS=${EPOCHS:-120}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
DROP_PATH_RATE=${DROP_PATH_RATE:-0.1}
TRAIN_RESIZE_MODE=${TRAIN_RESIZE_MODE:-random-resized-crop}
EVAL_RESIZE_MODE=${EVAL_RESIZE_MODE:-resize-center-crop}
NORM_MEAN=${NORM_MEAN:-0.5,0.5,0.5}
NORM_STD=${NORM_STD:-0.5,0.5,0.5}
VTAB_TRAIN_LIST=${VTAB_TRAIN_LIST:-train800val200.txt}
VTAB_VAL_LIST=${VTAB_VAL_LIST:-val200.txt}
VTAB_TEST_LIST=${VTAB_TEST_LIST:-test.txt}
RANK=${RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0}
BUDGET=${BUDGET:-0.2}
CATEGORY_RATIOS=${CATEGORY_RATIOS:-}
SELECTION_MODE=${SELECTION_MODE:-category}
UPDATE_INTERVAL=${UPDATE_INTERVAL:-10}
SENSITIVITY_BATCHES=${SENSITIVITY_BATCHES:-16}
SENSITIVITY_SCORE_MODE=${SENSITIVITY_SCORE_MODE:-grad_square}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-0}
STRONG_AUG=${STRONG_AUG:-0}
SPLIT_QKV_LORA=${SPLIT_QKV_LORA:-0}
SPT_SPARSE=${SPT_SPARSE:-0}
SPT_ALPHA=${SPT_ALPHA:-10}
SPT_BETA=${SPT_BETA:-5}

extra_args=()
if [[ "$STRONG_AUG" == "1" ]]; then
  extra_args+=(--strong-aug)
fi
if [[ "$SPLIT_QKV_LORA" == "1" ]]; then
  extra_args+=(--split-qkv-lora)
fi
if [[ "$SPT_SPARSE" == "1" ]]; then
  extra_args+=(--spt-sparse --spt-alpha "$SPT_ALPHA" --spt-beta "$SPT_BETA")
fi
if [[ -n "$CATEGORY_RATIOS" ]]; then
  extra_args+=(--category-ratios "$CATEGORY_RATIOS")
fi

command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: Python executable not found: $PYTHON" >&2; exit 1; }
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "error: invalid GPU_LIST: $GPU_LIST" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "error: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
TASKS=${TASKS-}
if [[ -z "${TASKS//[[:space:]]/}" ]]; then
  echo "error: TASKS must be set to a non-empty, space-separated list of VTAB task names" >&2
  exit 1
fi
read -ra TASK_ARRAY <<< "$TASKS"
missing_data=()
for task in "${TASK_ARRAY[@]}"; do
  task_root="$DATA_ROOT/vtab-1k/$task"
  for list_name in "$VTAB_TRAIN_LIST" "$VTAB_VAL_LIST" "$VTAB_TEST_LIST"; do
    [[ -f "$task_root/$list_name" ]] || missing_data+=("$task_root/$list_name")
  done
done
if (( ${#missing_data[@]} > 0 )); then
  echo "error: required VTAB files are missing:" >&2
  printf '  %s\n' "${missing_data[@]}" >&2
  exit 1
fi
if (( CHECK_ONLY == 1 )); then echo "preflight OK"; exit 0; fi

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
for task in "${TASK_ARRAY[@]}"; do
  output_dir="$OUTPUT_ROOT/$task"
  log_file="$LOG_ROOT/$task.log"
  if [[ ! -f "$output_dir/.complete" && ( -e "$output_dir" || -e "$log_file" ) ]]; then
    artifact_conflicts+=("$task")
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

for task in "${TASK_ARRAY[@]}"; do
  task_root="$DATA_ROOT/vtab-1k/$task"

  output_dir="$OUTPUT_ROOT/$task"
  log_file="$LOG_ROOT/${task}.log"
  if [[ -f "$output_dir/.complete" ]]; then
    echo "skip $task: success marker exists"
    continue
  fi
  if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "error: refusing existing run artifacts without .complete for $task" >&2
    echo "       choose a new OUTPUT_ROOT/LOG_ROOT or explicitly clean the old run artifacts" >&2
    exit 1
  fi

  gpu="${GPUS[$((job_index % gpu_count))]}"
  echo "launch $task on GPU $gpu -> $output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m daf_vit.train \
    --dataset "vtab:$task" \
    --data-root "$DATA_ROOT" \
    --output-dir "$output_dir" \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --method daf \
    --image-size 224 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --drop-path-rate "$DROP_PATH_RATE" \
    --train-resize-mode "$TRAIN_RESIZE_MODE" \
    --eval-resize-mode "$EVAL_RESIZE_MODE" \
    --vtab-train-list "$VTAB_TRAIN_LIST" \
    --vtab-val-list "$VTAB_VAL_LIST" \
    --vtab-test-list "$VTAB_TEST_LIST" \
    --rank "$RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --budget "$BUDGET" \
    --selection-mode "$SELECTION_MODE" \
    --update-interval "$UPDATE_INTERVAL" \
    --sensitivity-batches "$SENSITIVITY_BATCHES" \
    --sensitivity-score-mode "$SENSITIVITY_SCORE_MODE" \
    --norm-mean "$NORM_MEAN" \
    --norm-std "$NORM_STD" \
    --num-workers "$NUM_WORKERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --test-on-best-val-only \
    --device cuda \
    --amp \
    "${extra_args[@]}" > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("$task")
  JOB_OUTPUT_DIRS+=("$output_dir")

  job_index=$((job_index + 1))
  if (( job_index % gpu_count == 0 )); then
    if ! wait_for_jobs; then
      exit 1
    fi
  fi
done

if ! wait_for_jobs; then
  exit 1
fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$SCRIPT_DIR/summarize_table1_vtab.py" "$OUTPUT_ROOT" --expected-epochs "$EPOCHS"
