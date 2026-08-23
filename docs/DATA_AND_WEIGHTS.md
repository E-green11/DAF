# Data and ViT-B/16 Weight Layout

`DATA_ROOT` is the common directory containing CIFAR-100 and all VTAB-1k tasks.
Its default is the repository's `data/` directory. `CLEVR_DATA_ROOT` defaults
to `DATA_ROOT` and is only needed as an override when the two formal CLEVR
tasks live under another common root. `CHECKPOINT` defaults to
`checkpoints/ViT-B_16.npz` under the repository root.

## Checkpoint

Place a compatible ViT-B/16 NPZ checkpoint at:

```text
checkpoints/ViT-B_16.npz
```

The file is not distributed here. Keep the original filename and observe the
weight provider's license. To store it elsewhere, pass its absolute path:

```bash
CHECKPOINT=/path/to/ViT-B_16.npz bash scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```

## CIFAR-100

The launcher expects the extracted torchvision CIFAR-100 directory at:

```text
data/cifar-100-python/
```

With an external common root, the equivalent path is
`$DATA_ROOT/cifar-100-python/`.

## VTAB task directories

The complete workflow uses the following 18 VTAB task directories in addition
to CIFAR-100. Every row needs `train800val200.txt` and `test.txt`. The validation
list is `test.txt` for `caltech101` and `sun397`; every other row needs
`val200.txt`.

| Category | Task directory | Training list | Validation list | Test list |
| --- | --- | --- | --- | --- |
| Natural | `data/vtab-1k/caltech101/` | `train800val200.txt` | `test.txt` | `test.txt` |
| Natural | `data/vtab-1k/dtd/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Natural | `data/vtab-1k/oxford_flowers102/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Natural | `data/vtab-1k/oxford_iiit_pet/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Natural | `data/vtab-1k/svhn/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Natural | `data/vtab-1k/sun397/` | `train800val200.txt` | `test.txt` | `test.txt` |
| Specialized | `data/vtab-1k/eurosat/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Specialized | `data/vtab-1k/resisc45/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Specialized | `data/vtab-1k/diabetic_retinopathy/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Specialized | `data/vtab-1k/patch_camelyon/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/clevr_count/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/clevr_dist/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/dmlab/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/kitti/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/dsprites_loc/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/dsprites_ori/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/smallnorb_azi/` | `train800val200.txt` | `val200.txt` | `test.txt` |
| Structured | `data/vtab-1k/smallnorb_ele/` | `train800val200.txt` | `val200.txt` | `test.txt` |

Each nonempty list line has this form:

```text
relative/path/to/image.jpg INTEGER_LABEL
```

The image path is resolved relative to that task directory. Therefore the
paths written in all three lists must match the actual image tree exactly;
moving images without regenerating the lists will cause file-not-found errors.

The resulting default single-root layout is:

```text
data/
├── cifar-100-python/
└── vtab-1k/
    ├── caltech101/
    │   ├── train800val200.txt
    │   ├── test.txt
    │   └── <images referenced by the lists>
    ├── dtd/
    │   ├── train800val200.txt
    │   ├── val200.txt
    │   ├── test.txt
    │   └── <images referenced by the lists>
    ├── clevr_count/
    │   ├── train800val200.txt
    │   ├── val200.txt
    │   ├── test.txt
    │   └── <images referenced by the lists>
    ├── clevr_dist/
    │   ├── train800val200.txt
    │   ├── val200.txt
    │   ├── test.txt
    │   └── <images referenced by the lists>
    └── ...
```

## CLEVR source and split identity

The original `clevr_count` and `clevr_dist` runs used the CLEVR lists exported
through the SPT VTAB/TFDS pipeline, not the ordinary Hugging Face CLEVR export.
Those sources produce different example/label lists even when their directory
and split filenames look alike. Put the SPT/TFDS CLEVR export under
`$CLEVR_DATA_ROOT/vtab-1k`; by default `CLEVR_DATA_ROOT` equals `DATA_ROOT`.
Do not replace those two directories with the ordinary Hugging Face export.

[`../configs/clevr_spt_split_sha256.txt`](../configs/clevr_spt_split_sha256.txt)
records the SHA-256 values of exactly the six required CLEVR text split lists.
The launcher's preflight checks those six files to catch a substituted ordinary
export. The manifest covers split-list files only; image files are outside this
verification boundary.

## Preparation helpers

Together the helpers cover the workflow's 18 VTAB tasks: the Hugging Face
helper defaults to the 16 non-CLEVR tasks, while the SPT helper defaults to the
two CLEVR tasks in the same common root. Neither prepares CIFAR-100; install that
dataset separately under the common root. Set `TASKS="task_a task_b"` only to
override a helper's default subset.

The Hugging Face helper writes its 16 non-CLEVR tasks to `$DATA_ROOT/vtab-1k`
by default. It intentionally does not export CLEVR for this workflow:

```bash
bash scripts/prepare_vtab1k_hf.sh --check
bash scripts/prepare_vtab1k_hf.sh
```

It also accepts `OUTPUT_ROOT`, `DOWNLOAD_DIR`, `TASKS`, and `ZIP_PATH`.
Availability and licensing vary by source, so inspect the produced task trees
against the table above.

If an existing SPT checkout contains `data/vtab-source/get_vtab1k.py`, export
the two CLEVR tasks into the common root with:

```bash
SPT_ROOT=/path/to/SPT \
DATA_ROOT=/path/to/data \
bash scripts/prepare_vtab1k_spt.sh --check

SPT_ROOT=/path/to/SPT \
DATA_ROOT=/path/to/data \
bash scripts/prepare_vtab1k_spt.sh
```

Use the same SPT/TFDS source revision as the original split contract, then run
the main `--check`; a different revision or export is rejected if its lists do
not match the manifest. `TARGET_ROOT` can select another common root; set
`CLEVR_DATA_ROOT` to that same root when launching. Neither preparation script
downloads the ViT checkpoint.

## Validate without training

From any working directory, use absolute overrides when assets live elsewhere:

```bash
DATA_ROOT=/path/to/data \
CHECKPOINT=/path/to/ViT-B_16.npz \
GPU_LIST=0 \
bash /path/to/DAFgit/scripts/reproduce_table1_vit_b16_npz_test_best.sh --check
```

A successful check prints `preflight OK`. Missing paths are printed explicitly.
