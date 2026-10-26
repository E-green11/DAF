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

## CLEVR split-file contract

Put the required `clevr_count` and `clevr_dist` task directories under
`$CLEVR_DATA_ROOT/vtab-1k`; by default `CLEVR_DATA_ROOT` equals `DATA_ROOT`.
Their list files must match the manifest checked by the launcher.

[`../configs/clevr_spt_split_sha256.txt`](../configs/clevr_spt_split_sha256.txt)
records the SHA-256 values of exactly the six required CLEVR text split lists.
The launcher's preflight checks those files before training. The manifest covers
split-list files only; image files are outside this verification boundary.

