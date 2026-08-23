# Running the ViT-B/16 Workflow

The main entry point is:

```text
scripts/reproduce_table1_vit_b16_npz_test_best.sh
```

It launches the fixed CIFAR-100 plus 18-task VTAB workflow.
Run relative commands from the repository root. Once invoked, the launcher
resolves project paths from its own script location.

## Preflight

Review the interface and validate all external inputs before allocating a long
run:

```bash
bash scripts/reproduce_table1_vit_b16_npz_test_best.sh --help
bash scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```

From any other working directory, invoke the same preflight with an absolute
script path:

```bash
bash /path/to/DAFgit/scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```

`--check` does not train. It verifies the `PYTHON` executable, `GPU_LIST`
syntax, `CHECKPOINT`, `$DATA_ROOT/cifar-100-python`, and the task lists below
`$DATA_ROOT/vtab-1k`. An optional `CLEVR_DATA_ROOT` override can hold the two
CLEVR task directories elsewhere. Preflight also compares their six text split lists
with [`../configs/clevr_spt_split_sha256.txt`](../configs/clevr_spt_split_sha256.txt).
A mismatch stops before output directories or training processes are created.

## Full launch

One GPU runs the tasks sequentially:

```bash
GPU_LIST=0 bash scripts/reproduce_table1_vit_b16_npz_test_best.sh
```

Several GPUs run one task per listed GPU concurrently, in batches:

```bash
GPU_LIST=0,1,2,3 bash scripts/reproduce_table1_vit_b16_npz_test_best.sh
```

Each child process sees its assigned GPU through `CUDA_VISIBLE_DEVICES`. Do not
list a GPU that is unavailable or already too full for one training process.

## Configuration contract

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON` | `python` | Python executable or executable path |
| `DATA_ROOT` | `<repo>/data` | Common root containing `cifar-100-python` and all `vtab-1k` tasks |
| `CLEVR_DATA_ROOT` | `$DATA_ROOT` | Optional alternate common root for the two SPT VTAB/TFDS CLEVR task directories |
| `CHECKPOINT` | `<repo>/checkpoints/ViT-B_16.npz` | Compatible ViT-B/16 NPZ file |
| `OUTPUT_ROOT` | `<repo>/outputs/reproduce_table1_vit_b16_npz_test_best` | Parent directory for per-task artifacts |
| `LOG_ROOT` | `<repo>/logs/reproduce_table1_vit_b16_npz_test_best` | Parent directory for per-task text logs |
| `GPU_LIST` | `0` | Comma-separated physical GPU indices |
| `NUM_WORKERS` | `8` | DataLoader worker count per process |
| `BATCH_SIZE` | `64` | Per-process training batch size |
| `SEED` | `0` | Training and data-split seed |
| `MODEL` | `vit_base_patch16_224` | timm model identifier |
| `EVAL_INTERVAL` | `5` | Epoch interval between evaluations |
| `SAVE_EVERY` | `50` | Epoch interval between periodic checkpoints |

Example with all paths outside the repository:

```bash
PYTHON=python \
DATA_ROOT=/datasets/daf \
CHECKPOINT=/weights/ViT-B_16.npz \
OUTPUT_ROOT=/runs/daf-table1 \
LOG_ROOT=/runs/daf-table1-logs \
GPU_LIST=0,1,2,3 \
NUM_WORKERS=8 \
BATCH_SIZE=64 \
SEED=0 \
MODEL=vit_base_patch16_224 \
EVAL_INTERVAL=5 \
SAVE_EVERY=50 \
bash /path/to/DAFgit/scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```

Remove `--check` only after preflight succeeds.

The original CLEVR runs used the SPT VTAB/TFDS split semantics. An ordinary
Hugging Face CLEVR export has different lists and cannot replace this root.
The manifest verifies the three text lists for each of the two tasks; it does
not verify the referenced image files.

## Outputs and logs

Each task gets one directory and one log:

```text
$OUTPUT_ROOT/<task>/args.json
$OUTPUT_ROOT/<task>/metrics.jsonl
$OUTPUT_ROOT/<task>/best.pt
$OUTPUT_ROOT/<task>/epoch_<N>.pt
$OUTPUT_ROOT/<task>/.complete
$LOG_ROOT/<task>.log
```

`args.json` records the effective training arguments. `metrics.jsonl` contains
one JSON record per epoch, with evaluation fields on evaluation epochs.
Checkpoint files are generated runtime artifacts and can be large. Periodic
files follow `SAVE_EVERY`; `best.pt` is maintained by the trainer. After every
task finishes, the launcher prints a summary of the current run to standard
output. Redirect the launcher itself if that terminal output also needs to be
retained.

## Safe reruns and failures

The launcher uses two kinds of state:

- `$OUTPUT_ROOT/.daf-launch.lock` and `$LOG_ROOT/.daf-launch.lock` reserve those
  roots for one active launcher. Normal exit and handled interruption remove
  locks owned by that process. Never remove a lock while its launcher is alive.
- `$OUTPUT_ROOT/<task>/.complete` is written atomically only after that task
  process exits successfully. A rerun skips a task carrying this marker.

If a task fails or the machine stops abruptly, its output directory or log may
remain without `.complete`. The launcher refuses to overwrite such artifacts.
Inspect and archive that exact task directory and log, or choose fresh
`OUTPUT_ROOT` and `LOG_ROOT` values. If a hard-killed launcher left a lock,
first verify that no launcher still owns the roots; then remove only the exact
empty `.daf-launch.lock` directory with `rmdir`.

Training failure causes a nonzero launcher exit and is reported with the task
name. Other common preflight errors are fixed by choosing a working `PYTHON`, a
comma-separated numeric `GPU_LIST`, the correct `CHECKPOINT`, or the complete
data layout in [`DATA_AND_WEIGHTS.md`](DATA_AND_WEIGHTS.md).

## Reproducibility boundary

Keep both dataset roots, their lists, NPZ checkpoint, software versions,
command variables, and `SEED` unchanged when comparing reruns. Even on similar
servers, GPU kernels, drivers, multiprocessing order, and dependency
implementations may be nondeterministic, so bitwise identity is not guaranteed.
