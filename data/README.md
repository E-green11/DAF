# External datasets

This directory is a placeholder. The default runtime layout begins with:

```text
data/cifar-100-python/
data/vtab-1k/<task>/
```

Dataset files are intentionally excluded from the source release. Each VTAB
task requires images and launcher-specific split lists. The two CLEVR tasks
must use the original SPT VTAB/TFDS lists. `CLEVR_DATA_ROOT` defaults to
`DATA_ROOT` but can point at another common root; the six text-list SHA-256
values are recorded in
[`../configs/clevr_spt_split_sha256.txt`](../configs/clevr_spt_split_sha256.txt).
The manifest does not cover the referenced image files. See
[`../docs/DATA_AND_WEIGHTS.md`](../docs/DATA_AND_WEIGHTS.md) for all task names,
list filenames, preparation helpers, and external root usage.
