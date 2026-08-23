from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


def load_exporter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_vtab1k_hf.py"
    spec = importlib.util.spec_from_file_location("export_vtab1k_hf", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def use_one_item_splits(exporter, monkeypatch) -> None:
    monkeypatch.setitem(exporter.EXPECTED_TEST_COUNTS, "cifar", 1)
    for split in exporter.EXPECTED_SPLIT_COUNTS:
        monkeypatch.setitem(exporter.EXPECTED_SPLIT_COUNTS, split, 1)


def write_complete_task(task_root: Path, marker_name: str, *, marker: bool = True) -> None:
    for split in ("train800", "val200", "test", "train800val200"):
        image = task_root / "images" / split / "000000.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        (task_root / f"{split}.txt").write_text(f"images/{split}/000000.jpg 0\n")
    if marker:
        (task_root / marker_name).write_text("complete\n")


def write_task_zip(zip_path: Path, *, omit_image: str | None = None, truncate_split: str | None = None, unsafe_split: str | None = None) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for split in ("train800", "val200", "test", "train800val200"):
            rel_path = f"images/{split}/000000.jpg"
            if split == unsafe_split:
                rel_path = "../outside.jpg"
            list_text = "" if split == truncate_split else f"{rel_path} 0\n"
            archive.writestr(f"vtab-1k/cifar/{split}.txt", list_text)
            if split != omit_image:
                archive.writestr(f"vtab-1k/cifar/images/{split}/000000.jpg", b"image")


def test_member_task_and_relative_path_handles_wrapped_zip() -> None:
    exporter = load_exporter()
    matched = exporter.member_task_and_relative_path(
        "vtab-1k/cifar/images/train800/000001.jpg",
        {"cifar"},
    )
    assert matched == ("cifar", Path("images/train800/000001.jpg"))


def test_member_task_and_relative_path_rejects_traversal() -> None:
    exporter = load_exporter()
    with pytest.raises(ValueError, match="unsafe archive member"):
        exporter.member_task_and_relative_path("vtab-1k/cifar/../../escape", {"cifar"})


def test_task_is_complete_requires_marker_and_all_images(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER, marker=False)
    assert not exporter.task_is_complete(tmp_path, "cifar")

    (task_root / exporter.EXPORT_MARKER).write_text("complete\n")
    (task_root / "images" / "test" / "000000.jpg").unlink()
    assert not exporter.task_is_complete(tmp_path, "cifar")


def test_task_is_complete_rejects_unsafe_list_path(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER)
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    (task_root / "test.txt").write_text("../outside.jpg 0\n")
    assert not exporter.task_is_complete(tmp_path, "cifar")


def test_extract_tasks_skips_only_marked_complete_task(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    output_root = tmp_path / "vtab-1k"
    task_root = output_root / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER)
    zip_path = tmp_path / "vtab-1k.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("vtab-1k/cifar/train800.txt", "bad\n")

    exporter.extract_tasks(zip_path, output_root, ["cifar"], skip_existing=True)

    assert (task_root / "train800.txt").read_text() == "images/train800/000000.jpg 0\n"


@pytest.mark.parametrize(
    ("zip_kwargs", "message"),
    [
        ({"omit_image": "test"}, "required files/counts are incomplete"),
        ({"truncate_split": "val200"}, "required files/counts are incomplete"),
        ({"unsafe_split": "train800"}, "required files/counts are incomplete"),
    ],
)
def test_extract_tasks_rejects_invalid_task_without_publishing(tmp_path, monkeypatch, zip_kwargs, message) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    output_root = tmp_path / "vtab-1k"
    zip_path = tmp_path / "vtab-1k.zip"
    write_task_zip(zip_path, **zip_kwargs)

    with pytest.raises(RuntimeError, match=message):
        exporter.extract_tasks(zip_path, output_root, ["cifar"], skip_existing=True)

    assert not (output_root / "cifar").exists()
    assert not list(output_root.glob(".cifar.export-*"))


def test_extract_tasks_publishes_marker_after_full_validation(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    output_root = tmp_path / "vtab-1k"
    zip_path = tmp_path / "vtab-1k.zip"
    write_task_zip(zip_path)

    exporter.extract_tasks(zip_path, output_root, ["cifar"], skip_existing=True)

    assert exporter.task_is_complete(output_root, "cifar")
    assert (output_root / "cifar" / exporter.EXPORT_MARKER).read_text() == "complete\n"


def test_extract_tasks_refuses_to_mix_with_existing_partial_target(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    use_one_item_splits(exporter, monkeypatch)
    output_root = tmp_path / "vtab-1k"
    task_root = output_root / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER, marker=False)
    zip_path = tmp_path / "vtab-1k.zip"
    write_task_zip(zip_path)

    with pytest.raises(RuntimeError, match="existing incomplete target"):
        exporter.extract_tasks(zip_path, output_root, ["cifar"], skip_existing=True)

    assert not (task_root / exporter.EXPORT_MARKER).exists()
    assert (task_root / "train800.txt").exists()
