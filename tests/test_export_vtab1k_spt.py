from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_exporter():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_vtab1k_spt.py"
    spec = importlib.util.spec_from_file_location("export_vtab1k_spt", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_complete_task(task_root: Path, marker_name: str, *, omit_split: str | None = None) -> None:
    for split in ("train800", "val200", "test", "train800val200"):
        if split == omit_split:
            continue
        image = task_root / "images" / split / "000000.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        (task_root / f"{split}.txt").write_text(f"images/{split}/000000.jpg 0\n")
    (task_root / marker_name).write_text("complete\n")


class FakeValue:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self

    def item(self):
        return self.value


class FakeImageValue:
    def numpy(self):
        return [[0]]


class FakeDataset:
    def __init__(self, fail: bool):
        self.fail = fail

    def map(self, _preprocess_fn, _threads):
        return self

    def __iter__(self):
        if self.fail:
            raise RuntimeError("interrupted export")
        yield {"label": FakeValue(0), "image": FakeImageValue()}


class FakeData:
    _image_decoder = object()
    _base_preprocess_fn = object()
    _num_preprocessing_threads = 1

    def __init__(self, fail_split: str | None):
        self.fail_split = fail_split

    def _get_dataset_split(self, split_name: str, shuffle_files: bool):
        assert shuffle_files is False
        return FakeDataset(split_name == self.fail_split)


class FakeRegistry:
    def __init__(self, fail_split: str | None):
        self.fail_split = fail_split

    def lookup(self, _name: str):
        return lambda **_params: FakeData(self.fail_split)


class FakeSavedImage:
    def save(self, path: Path) -> None:
        path.write_bytes(b"image")


def configure_fake_export(exporter, monkeypatch) -> None:
    monkeypatch.setattr(exporter.importlib, "import_module", lambda _name: None)
    monkeypatch.setattr(exporter.Image, "fromarray", lambda _array: FakeSavedImage())


def test_task_is_complete_requires_all_four_lists_and_marker(tmp_path) -> None:
    exporter = load_exporter()
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER, omit_split="train800val200")
    assert not exporter.task_is_complete(tmp_path, "cifar")

    image = task_root / "images" / "train800val200" / "000000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    (task_root / "train800val200.txt").write_text("images/train800val200/000000.jpg 0\n")
    assert exporter.task_is_complete(tmp_path, "cifar")


def test_task_is_complete_rejects_missing_image_and_traversal(tmp_path) -> None:
    exporter = load_exporter()
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER)
    (task_root / "images" / "test" / "000000.jpg").unlink()
    assert not exporter.task_is_complete(tmp_path, "cifar")

    (tmp_path / "outside.jpg").write_bytes(b"outside")
    (task_root / "test.txt").write_text("../outside.jpg 0\n")
    assert not exporter.task_is_complete(tmp_path, "cifar")


def test_export_task_skips_only_fully_complete_target(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER)
    configure_fake_export(exporter, monkeypatch)

    class RejectRegistry:
        @staticmethod
        def lookup(_name):
            raise AssertionError("a complete task must be skipped before dataset creation")

    exporter.export_task("cifar", tmp_path, RejectRegistry, lambda *_args: None, skip_existing=True)


def test_interrupted_export_leaves_no_target_and_can_retry(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    configure_fake_export(exporter, monkeypatch)

    with pytest.raises(RuntimeError, match="interrupted export"):
        exporter.export_task("cifar", tmp_path, FakeRegistry("val200"), lambda *_args: None, skip_existing=True)

    assert not (tmp_path / "cifar").exists()
    assert not list(tmp_path.glob(".cifar.export-*"))

    exporter.export_task("cifar", tmp_path, FakeRegistry(None), lambda *_args: None, skip_existing=True)
    assert exporter.task_is_complete(tmp_path, "cifar")


def test_dataset_initialization_failure_cleans_staging(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    monkeypatch.setattr(
        exporter.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("initialization failed")),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        exporter.export_task("cifar", tmp_path, FakeRegistry(None), lambda *_args: None, skip_existing=True)

    assert not (tmp_path / "cifar").exists()
    assert not list(tmp_path.glob(".cifar.export-*"))


def test_export_task_refuses_existing_partial_target(tmp_path, monkeypatch) -> None:
    exporter = load_exporter()
    task_root = tmp_path / "cifar"
    write_complete_task(task_root, exporter.EXPORT_MARKER, omit_split="train800val200")
    configure_fake_export(exporter, monkeypatch)

    with pytest.raises(RuntimeError, match="existing incomplete target"):
        exporter.export_task("cifar", tmp_path, FakeRegistry(None), lambda *_args: None, skip_existing=True)

    assert task_root.exists()
