from __future__ import annotations

import argparse
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daf_vit.data import VTAB_TASKS, get_vtab_task  # noqa: E402


REQUIRED_SPLITS = ("train800", "val200", "test", "train800val200")
EXPORT_MARKER = ".export-complete"
EXPECTED_TEST_COUNTS = {
    "caltech101": 6084,
    "cifar": 10000,
    "clevr_count": 15000,
    "clevr_dist": 15000,
    "diabetic_retinopathy": 42670,
    "dmlab": 22735,
    "dsprites_loc": 73728,
    "dsprites_ori": 73728,
    "dtd": 1880,
    "eurosat": 5400,
    "kitti": 711,
    "oxford_flowers102": 6149,
    "oxford_iiit_pet": 3669,
    "patch_camelyon": 32768,
    "resisc45": 6300,
    "smallnorb_azi": 12150,
    "smallnorb_ele": 12150,
    "sun397": 21750,
    "svhn": 26032,
}
EXPECTED_SPLIT_COUNTS = {
    "train800": 800,
    "val200": 200,
    "train800val200": 1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download/extract the preprocessed VTAB-1K zip from Hugging Face")
    parser.add_argument("--repo-id", default="XiN0919/VTAB-1k")
    parser.add_argument("--filename", default="vtab-1k.zip")
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--zip-path", default="")
    parser.add_argument("--output-root", default="data/vtab-1k")
    parser.add_argument("--tasks", nargs="+", default=[task.name for task in VTAB_TASKS])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def canonical_tasks(tasks: list[str]) -> list[str]:
    return [get_vtab_task(task).name for task in tasks]


def expected_count(task: str, split: str) -> int:
    if split == "test":
        return EXPECTED_TEST_COUNTS[task]
    return EXPECTED_SPLIT_COUNTS[split]


def _safe_image_path(task_root: Path, value: str) -> Path | None:
    relative = Path(value)
    if not value or "\\" in value or relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = task_root / relative
    try:
        if not candidate.is_file():
            return None
        candidate.resolve(strict=True).relative_to(task_root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return candidate


def _task_contents_are_complete(task_root: Path, task: str) -> bool:
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
                    if _safe_image_path(task_root, image_path) is None:
                        return False
                    count += 1
        except (OSError, ValueError):
            return False
        if count != expected_count(task, split):
            return False
    return True


def task_is_complete(root: Path, task: str) -> bool:
    task_root = root / task
    return (task_root / EXPORT_MARKER).is_file() and _task_contents_are_complete(task_root, task)


def _write_export_marker(task_root: Path) -> None:
    temporary = task_root / f"{EXPORT_MARKER}.tmp"
    temporary.write_text("complete\n")
    temporary.replace(task_root / EXPORT_MARKER)


def member_task_and_relative_path(member_name: str, task_names: set[str]) -> tuple[str, Path] | None:
    if "\\" in member_name:
        if task_names.intersection(member_name.split("/")):
            raise ValueError(f"unsafe archive member: {member_name}")
        return None
    pure_path = PurePosixPath(member_name)
    parts = [part for part in pure_path.parts if part not in ("", ".", "/")]
    if not parts or parts[0] == "__MACOSX":
        return None
    for index, part in enumerate(parts):
        if part in task_names and index + 1 < len(parts):
            relative_parts = parts[index + 1 :]
            if pure_path.is_absolute() or ".." in relative_parts:
                raise ValueError(f"unsafe archive member: {member_name}")
            return part, Path(*relative_parts)
    return None


def download_zip(repo_id: str, filename: str, download_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    download_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=download_dir,
        )
    )


def extract_tasks(zip_path: Path, output_root: Path, tasks: list[str], skip_existing: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    task_names = set(tasks)
    extracted = {task: 0 for task in tasks}
    active_tasks: list[str] = []
    for task in tasks:
        task_root = output_root / task
        if skip_existing and task_is_complete(output_root, task):
            print(f"{task}: already complete, skipping")
            continue
        if task_root.exists():
            raise RuntimeError(f"{task}: existing incomplete target; move it aside before retrying")
        active_tasks.append(task)

    staging_roots: dict[str, Path] = {}
    published: set[str] = set()
    try:
        for task in active_tasks:
            staging_roots[task] = Path(tempfile.mkdtemp(prefix=f".{task}.export-", dir=output_root))

        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                matched = member_task_and_relative_path(info.filename, set(active_tasks))
                if matched is None:
                    continue
                task, relative_path = matched
                if info.is_dir():
                    continue
                file_type = (info.external_attr >> 16) & 0o170000
                if stat.S_ISLNK(file_type):
                    raise ValueError(f"unsafe archive member: {info.filename}")
                destination = staging_roots[task] / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                extracted[task] += 1

        failures: list[str] = []
        for task in active_tasks:
            if extracted[task] == 0:
                failures.append(f"{task}: no files found in {zip_path}")
            elif not _task_contents_are_complete(staging_roots[task], task):
                failures.append(f"{task}: extracted but required files/counts are incomplete")
        if failures:
            raise RuntimeError("; ".join(failures))

        for task in active_tasks:
            staging_root = staging_roots[task]
            _write_export_marker(staging_root)
            staging_root.replace(output_root / task)
            published.add(task)
            print(f"{task}: extracted {extracted[task]} files")
    finally:
        for task, staging_root in staging_roots.items():
            if task not in published and staging_root.exists():
                shutil.rmtree(staging_root)


def main() -> None:
    args = parse_args()
    tasks = canonical_tasks(args.tasks)
    output_root = Path(args.output_root).resolve()
    if args.zip_path:
        zip_path = Path(args.zip_path).resolve()
    else:
        zip_path = download_zip(args.repo_id, args.filename, Path(args.download_dir).resolve())
    extract_tasks(zip_path, output_root, tasks, args.skip_existing)


if __name__ == "__main__":
    main()
