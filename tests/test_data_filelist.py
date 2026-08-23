from pathlib import Path

from PIL import Image

from daf_vit.data import FileListDataset, build_dataset_bundle, build_eval_transform, get_vtab_task


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(path)


def _write_vtab_split(root: Path, split: str, labels: list[int]) -> None:
    lines = []
    for index, label in enumerate(labels):
        image_path = f"images/{split}/{index:06d}.jpg"
        _write_image(root / image_path)
        lines.append(f"{image_path} {label}\n")
    (root / f"{split}.txt").write_text("".join(lines))


def test_file_list_dataset_reads_relative_paths_and_labels(tmp_path):
    dataset_root = tmp_path / "vtab-1k" / "caltech101"
    _write_vtab_split(dataset_root, "train800", [3, 7])

    dataset = FileListDataset(dataset_root, "train800.txt", transform=None)

    image, label = dataset[1]
    assert image.size == (8, 8)
    assert label == 7
    assert len(dataset) == 2


def test_build_dataset_bundle_loads_spt_vtab_layout(tmp_path):
    dataset_root = tmp_path / "vtab-1k" / "caltech101"
    _write_vtab_split(dataset_root, "train800", [0, 1])
    _write_vtab_split(dataset_root, "val200", [2])
    _write_vtab_split(dataset_root, "test", [3, 4, 5])

    bundle = build_dataset_bundle(
        dataset="caltech101",
        root=tmp_path,
        seed=0,
        image_size=32,
        download=False,
        strong_aug=False,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    assert len(bundle.train) == 2
    assert len(bundle.val) == 1
    assert len(bundle.test) == 3
    assert bundle.num_classes == 102


def test_build_dataset_bundle_can_use_train800val200_for_spt_test_protocol(tmp_path):
    dataset_root = tmp_path / "vtab-1k" / "caltech101"
    _write_vtab_split(dataset_root, "train800", [0, 1])
    _write_vtab_split(dataset_root, "val200", [2])
    _write_vtab_split(dataset_root, "train800val200", [0, 1, 2])
    _write_vtab_split(dataset_root, "test", [3, 4, 5])

    bundle = build_dataset_bundle(
        dataset="caltech101",
        root=tmp_path,
        seed=0,
        image_size=32,
        download=False,
        strong_aug=False,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        vtab_train_list="train800val200.txt",
        vtab_val_list="test.txt",
        vtab_test_list="test.txt",
    )

    assert len(bundle.train) == 3
    assert len(bundle.val) == 3
    assert len(bundle.test) == 3


def test_get_vtab_task_accepts_aliases():
    assert get_vtab_task("cifar").name == "cifar"
    assert get_vtab_task("cifar100").name == "cifar"
    assert get_vtab_task("smallnorb-ele").name == "smallnorb_ele"


def test_eval_transform_can_use_direct_resize():
    transform = build_eval_transform(
        image_size=32,
        resize_mode="direct-resize",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    names = [op.__class__.__name__ for op in transform.transforms]

    assert names[:2] == ["Resize", "CenterCrop"]
    assert tuple(transform.transforms[0].size) == (32, 32)
