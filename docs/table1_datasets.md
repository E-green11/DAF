# Dataset Plan for the Reproduction Workflow

The workflow uses one torchvision CIFAR-100 directory and 18 VTAB task
directories. The canonical placement and list-format instructions are in
[`DATA_AND_WEIGHTS.md`](DATA_AND_WEIGHTS.md).

## Required roots

```text
data/cifar-100-python/
data/vtab-1k/<task>/
```

Every VTAB task needs `train800val200.txt` and `test.txt`. The validation list
is `test.txt` for `caltech101` and `sun397`; it is `val200.txt` for the other 16
tasks.

| Category | Task directories |
| --- | --- |
| Natural | `caltech101`, `dtd`, `oxford_flowers102`, `oxford_iiit_pet`, `svhn`, `sun397` |
| Specialized | `eurosat`, `resisc45`, `diabetic_retinopathy`, `patch_camelyon` |
| Structured | `clevr_count`, `clevr_dist`, `dmlab`, `kitti`, `dsprites_loc`, `dsprites_ori`, `smallnorb_azi`, `smallnorb_ele` |

List entries contain an image path relative to the corresponding task
directory followed by an integer label. The image files may use any directory
layout, provided every path in the lists points to the actual file.

## Preparation and validation

The preparation helpers cover all 18 VTAB task directories: the Hugging Face
helper prepares 16 tasks, while the optional CLEVR helper prepares the two
manifest-checked CLEVR task directories under the same common root:

```bash
bash scripts/prepare_vtab1k_hf.sh --check
bash scripts/prepare_vtab1k_hf.sh
```

For an existing compatible CLEVR preparation checkout:

```bash
SPT_ROOT=/path/to/SPT bash scripts/prepare_vtab1k_spt.sh --check
SPT_ROOT=/path/to/SPT bash scripts/prepare_vtab1k_spt.sh
```

The optional CLEVR preparation helper uses a separate Python 3.10 compatibility
environment. Install `requirements-spt.txt` there; do not add its TensorFlow and
TensorFlow Addons dependencies to the main DAF training environment. The
Hugging Face helper uses the main environment.

Both exporters publish one task at a time only after all four lists and every
referenced image have been validated. A published task has an
`.export-complete` marker. `--skip-existing` trusts only that marker together
with a fresh full validation. If a task directory exists without satisfying
that contract, move it aside before retrying; the exporters do not merge into
or delete partial data.

Use `TASKS="task_a task_b"` only when intentionally preparing a subset. After
placing CIFAR-100, all VTAB files, and the checkpoint, validate the complete
layout without starting training:

```bash
bash scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```
