from __future__ import annotations

import argparse
import importlib
import os
import resource
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image


TASK_CONFIGS = {
    "cifar": ("cifar", {"num_classes": 100}),
    "caltech101": ("caltech101", {}),
    "dtd": ("dtd", {}),
    "oxford_flowers102": ("oxford_flowers102", {}),
    "oxford_iiit_pet": ("oxford_iiit_pet", {}),
    "svhn": ("svhn", {}),
    "sun397": ("sun397", {}),
    "eurosat": ("eurosat", {}),
    "resisc45": ("resisc45", {}),
    "patch_camelyon": ("patch_camelyon", {}),
    "diabetic_retinopathy": ("diabetic_retinopathy", {"config": "btgraham-300"}),
    "clevr_count": ("clevr", {"task": "count_all"}),
    "clevr_dist": ("clevr", {"task": "closest_object_distance"}),
    "dmlab": ("dmlab", {}),
    "kitti": ("kitti", {"task": "closest_vehicle_distance"}),
    "dsprites_loc": ("dsprites", {"predicted_attribute": "label_x_position", "num_classes": 16}),
    "dsprites_ori": ("dsprites", {"predicted_attribute": "label_orientation", "num_classes": 16}),
    "smallnorb_azi": ("smallnorb", {"predicted_attribute": "label_azimuth"}),
    "smallnorb_ele": ("smallnorb", {"predicted_attribute": "label_elevation"}),
}
REQUIRED_SPLITS = ("train800", "val200", "test", "train800val200")
EXPORT_MARKER = ".export-complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export SPT/VTAB-1k datasets to image-list folders")
    parser.add_argument("--spt-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _prepare_spt_imports(spt_root: Path):
    source_root = spt_root / "data" / "vtab-source"
    if not source_root.exists():
        raise FileNotFoundError(f"missing SPT VTAB source: {source_root}")
    (source_root / "data" / "vtab").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)

    from task_adaptation.data.base import compose_preprocess_fn
    from task_adaptation.registry import Registry

    return Registry, compose_preprocess_fn


def _raise_file_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def _safe_image_exists(task_root: Path, value: str) -> bool:
    relative = Path(value)
    if not value or "\\" in value or relative.is_absolute() or ".." in relative.parts:
        return False
    candidate = task_root / relative
    try:
        if not candidate.is_file():
            return False
        candidate.resolve(strict=True).relative_to(task_root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _task_contents_are_complete(task_root: Path) -> bool:
    for split in REQUIRED_SPLITS:
        list_path = task_root / f"{split}.txt"
        if not list_path.is_file():
            return False
        count = 0
        try:
            with list_path.open() as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    fields = stripped.rsplit(maxsplit=1)
                    if len(fields) != 2:
                        return False
                    image_path, label = fields
                    int(label)
                    if not _safe_image_exists(task_root, image_path):
                        return False
                    count += 1
        except (OSError, ValueError):
            return False
        if count == 0:
            return False
    return True


def task_is_complete(root: Path, task: str) -> bool:
    task_root = root / task
    return (task_root / EXPORT_MARKER).is_file() and _task_contents_are_complete(task_root)


def _write_export_marker(task_root: Path) -> None:
    temporary = task_root / f"{EXPORT_MARKER}.tmp"
    temporary.write_text("complete\n")
    temporary.replace(task_root / EXPORT_MARKER)


def export_task(task_name: str, output_root: Path, registry, compose_preprocess_fn, skip_existing: bool) -> None:
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"unknown VTAB task: {task_name}")
    registry_name, params = TASK_CONFIGS[task_name]
    task_root = output_root / task_name
    if skip_existing and task_is_complete(output_root, task_name):
        print(f"{task_name}: already complete, skipping")
        return
    if task_root.exists():
        raise RuntimeError(f"{task_name}: existing incomplete target; move it aside before retrying")

    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{task_name}.export-", dir=output_root))
    published = False
    try:
        importlib.import_module(f"task_adaptation.data.{registry_name}")
        data_cls = registry.lookup(f"data.{registry_name}")(**dict(params))
        print(f"{task_name}: exporting")
        for split_name in REQUIRED_SPLITS:
            data = data_cls._get_dataset_split(split_name=split_name, shuffle_files=False)
            preprocess_fn = compose_preprocess_fn(data_cls._image_decoder, data_cls._base_preprocess_fn)
            data = data.map(preprocess_fn, data_cls._num_preprocessing_threads)

            split_dir = staging_root / "images" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            list_path = staging_root / f"{split_name}.txt"
            with list_path.open("w") as handle:
                for index, item in enumerate(data):
                    rel_path = Path("images") / split_name / f"{index:06d}.jpg"
                    label = int(item["label"].numpy().item())
                    image = Image.fromarray(item["image"].numpy())
                    image.save(staging_root / rel_path)
                    handle.write(f"{rel_path.as_posix()} {label}\n")
            print(f"{task_name}: wrote {split_name}")
        if not _task_contents_are_complete(staging_root):
            raise RuntimeError(f"{task_name}: export validation failed")
        _write_export_marker(staging_root)
        staging_root.replace(task_root)
        published = True
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)


def main() -> None:
    args = parse_args()
    _raise_file_limit()
    output_root = Path(args.output_root).resolve()
    registry, compose_preprocess_fn = _prepare_spt_imports(Path(args.spt_root).resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    for task in args.tasks:
        try:
            export_task(task, output_root, registry, compose_preprocess_fn, args.skip_existing)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((task, repr(exc)))
            print(f"{task}: failed: {exc!r}", file=sys.stderr)
    if failures:
        failure_path = output_root / "export_failures.txt"
        with failure_path.open("w") as handle:
            for task, error in failures:
                handle.write(f"{task}\t{error}\n")
        print(f"wrote {failure_path}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
