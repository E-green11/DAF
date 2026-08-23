#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/checkpoints/ViT-B_16.npz}
MODEL=${MODEL:-vit_base_patch16_224}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/tune_structured_testbest_stage1}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/logs/tune_structured_testbest_stage1}
GPU_LIST=${GPU_LIST-0}

usage() {
  cat <<'EOF'
Usage: bash scripts/tune_structured_testbest_stage1.sh [--check|--help]

Run the retained structured-task tuning configurations. Override PYTHON,
DATA_ROOT, CHECKPOINT, OUTPUT_ROOT, LOG_ROOT, GPU_LIST, NUM_WORKERS,
BATCH_SIZE, SEED, or the schedule variables in this file.
EOF
}

CHECK_ONLY=0
case "${1:-}" in
  "") ;;
  --check) CHECK_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-64}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
NUM_WORKERS=${NUM_WORKERS:-8}
SAVE_EVERY=${SAVE_EVERY:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
SEED=${SEED:-0}

command -v "$PYTHON" >/dev/null 2>&1 || { echo "error: Python executable not found: $PYTHON" >&2; exit 1; }
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "error: invalid GPU_LIST: $GPU_LIST" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "error: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
missing_data=()
for task in dsprites_loc dsprites_ori clevr_dist; do
  task_root="$DATA_ROOT/vtab-1k/$task"
  for list_name in train800val200.txt val200.txt test.txt; do
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
RUN_NAMES=(
  dsprites_loc_direct_cat_gs_lr1e3
  dsprites_ori_direct_cat_gs_lr1e3
  clevr_dist_split_global_gwa_lr1e3
)

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

launch() {
  local name="$1"
  local task="$2"
  local lr="$3"
  local label_smoothing="$4"
  local drop_path="$5"
  local rank="$6"
  local alpha="$7"
  local lora_dropout="$8"
  local selection_mode="$9"
  local score_mode="${10}"
  local split_qkv="${11}"
  local update_interval="${12}"
  local sensitivity_batches="${13}"

  local output_dir="$OUTPUT_ROOT/$name"
  local log_file="$LOG_ROOT/$name.log"
  if [[ -f "$output_dir/.complete" ]]; then
    echo "skip $name: success marker exists"
    return 0
  fi
  if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "error: refusing existing run artifacts without .complete for $name" >&2
    echo "       choose a new OUTPUT_ROOT/LOG_ROOT or explicitly clean the old run artifacts" >&2
    return 1
  fi

  local gpu="${GPUS[$((job_index % gpu_count))]}"
  echo "launch $name ($task) on GPU $gpu -> $output_dir"
  local extra_args=()
  if [[ "$split_qkv" == "1" ]]; then
    extra_args+=(--split-qkv-lora)
  fi

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
    --lr "$lr" \
    --weight-decay "$WEIGHT_DECAY" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --label-smoothing "$label_smoothing" \
    --drop-path-rate "$drop_path" \
    --train-resize-mode direct-resize \
    --eval-resize-mode direct-resize \
    --vtab-train-list train800val200.txt \
    --vtab-val-list val200.txt \
    --vtab-test-list test.txt \
    --rank "$rank" \
    --lora-alpha "$alpha" \
    --lora-dropout "$lora_dropout" \
    --budget 0.2 \
    --selection-mode "$selection_mode" \
    --update-interval "$update_interval" \
    --sensitivity-batches "$sensitivity_batches" \
    --sensitivity-score-mode "$score_mode" \
    --norm-mean 0.5,0.5,0.5 \
    --norm-std 0.5,0.5,0.5 \
    --num-workers "$NUM_WORKERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --save-every "$SAVE_EVERY" \
    --device cuda \
    --amp \
    "${extra_args[@]}" > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("$name")
  JOB_OUTPUT_DIRS+=("$output_dir")

  job_index=$((job_index + 1))
  if (( job_index % gpu_count == 0 )); then
    if ! wait_for_jobs; then
      return 1
    fi
  fi
}

launch dsprites_loc_direct_cat_gs_lr1e3 dsprites_loc 0.001 0.0 0.0 8 16 0.0 category grad_square 0 10 16
launch dsprites_ori_direct_cat_gs_lr1e3 dsprites_ori 0.001 0.0 0.0 8 16 0.0 category grad_square 0 10 16
launch clevr_dist_split_global_gwa_lr1e3 clevr_dist 0.001 0.0 0.1 8 8 0.1 global grad_weight_abs 1 10 16

if ! wait_for_jobs; then
  exit 1
fi
