from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    num_classes: int


@dataclass(frozen=True)
class VTABTask:
    name: str
    num_classes: int
    category: str
    aliases: tuple[str, ...] = ()


VTAB_TASKS: tuple[VTABTask, ...] = (
    VTABTask("cifar", 100, "natural", aliases=("cifar100", "cifar-100")),
    VTABTask("caltech101", 102, "natural"),
    VTABTask("dtd", 47, "natural"),
    VTABTask("oxford_flowers102", 102, "natural", aliases=("flowers102", "flower102")),
    VTABTask("oxford_iiit_pet", 37, "natural", aliases=("pets", "oxford-pets")),
    VTABTask("svhn", 10, "natural"),
    VTABTask("sun397", 397, "natural"),
    VTABTask("eurosat", 10, "specialized"),
    VTABTask("resisc45", 45, "specialized"),
    VTABTask("patch_camelyon", 2, "specialized", aliases=("camelyon",)),
    VTABTask("diabetic_retinopathy", 5, "specialized", aliases=("retinopathy",)),
    VTABTask("clevr_count", 8, "structured", aliases=("clevr-count",)),
    VTABTask("clevr_dist", 6, "structured", aliases=("clevr-dist",)),
    VTABTask("dmlab", 6, "structured"),
    VTABTask("kitti", 4, "structured", aliases=("kitti-dist",)),
    VTABTask("dsprites_loc", 16, "structured", aliases=("dsprites-loc",)),
    VTABTask("dsprites_ori", 16, "structured", aliases=("dsprites-ori",)),
    VTABTask("smallnorb_azi", 18, "structured", aliases=("smallnorb-azi", "smallnorb-azim")),
    VTABTask("smallnorb_ele", 9, "structured", aliases=("smallnorb-ele",)),
)


def _canonical_key(value: str) -> str:
    return value.lower().replace("vtab:", "").replace("-", "_")


def get_vtab_task(dataset: str) -> VTABTask:
    key = _canonical_key(dataset)
    for task in VTAB_TASKS:
        if key == task.name or key in {_canonical_key(alias) for alias in task.aliases}:
            return task
    raise ValueError(f"unsupported VTAB task: {dataset}")


class FileListDataset(Dataset):
    def __init__(self, root: str | Path, list_file: str, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        path = self.root / list_file
        if not path.exists():
            raise FileNotFoundError(f"missing dataset split file: {path}")
        with path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rel_path, label = stripped.rsplit(maxsplit=1)
                self.samples.append((self.root / rel_path, int(label)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_train_val_indices(dataset_size: int, train_size: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    if train_size + val_size > dataset_size:
        raise ValueError("train_size + val_size must be <= dataset_size")
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:train_size], indices[train_size : train_size + val_size]


def build_train_transform(
    image_size: int,
    strong_aug: bool,
    resize_mode: str = "random-resized-crop",
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    if resize_mode in {"random-resized-crop", "random-resized-crop-nohflip"}:
        ops: list = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
        if resize_mode == "random-resized-crop":
            ops.append(transforms.RandomHorizontalFlip())
    elif resize_mode == "direct-resize":
        ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    elif resize_mode == "direct-resize-hflip":
        ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise ValueError(f"unsupported train resize mode: {resize_mode}")
    if strong_aug:
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(
    image_size: int,
    resize_mode: str = "resize-center-crop",
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    if resize_mode == "resize-center-crop":
        resize_size = max(image_size, int(image_size / 0.875))
        ops: list = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]
    elif resize_mode == "direct-resize":
        ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"unsupported eval resize mode: {resize_mode}")
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


def build_cifar100_vtab(
    root: str | Path,
    seed: int,
    image_size: int = 224,
    train_size: int = 800,
    val_size: int = 200,
    download: bool = True,
    strong_aug: bool = False,
    train_resize_mode: str = "random-resized-crop",
    eval_resize_mode: str = "resize-center-crop",
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> DatasetBundle:
    root = Path(root)
    train_full = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=build_train_transform(
            image_size,
            strong_aug=strong_aug,
            resize_mode=train_resize_mode,
            mean=mean,
            std=std,
        ),
        download=download,
    )
    val_full = datasets.CIFAR100(
        root=str(root),
        train=True,
        transform=build_eval_transform(image_size, resize_mode=eval_resize_mode, mean=mean, std=std),
        download=download,
    )
    test = datasets.CIFAR100(
        root=str(root),
        train=False,
        transform=build_eval_transform(image_size, resize_mode=eval_resize_mode, mean=mean, std=std),
        download=download,
    )
    train_idx, val_idx = make_train_val_indices(len(train_full), train_size, val_size, seed)
    return DatasetBundle(
        train=Subset(train_full, train_idx),
        val=Subset(val_full, val_idx),
        test=test,
        num_classes=100,
    )


def build_spt_vtab(
    root: str | Path,
    dataset: str,
    image_size: int = 224,
    strong_aug: bool = False,
    train_resize_mode: str = "random-resized-crop",
    eval_resize_mode: str = "resize-center-crop",
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    train_list: str = "train800.txt",
    val_list: str = "val200.txt",
    test_list: str = "test.txt",
) -> DatasetBundle:
    task = get_vtab_task(dataset)
    base = Path(root)
    dataset_root = base / "vtab-1k" / task.name
    if not dataset_root.exists():
        dataset_root = base / task.name
    train = FileListDataset(
        dataset_root,
        train_list,
        transform=build_train_transform(
            image_size,
            strong_aug=strong_aug,
            resize_mode=train_resize_mode,
            mean=mean,
            std=std,
        ),
    )
    val = FileListDataset(
        dataset_root,
        val_list,
        transform=build_eval_transform(image_size, resize_mode=eval_resize_mode, mean=mean, std=std),
    )
    test = FileListDataset(
        dataset_root,
        test_list,
        transform=build_eval_transform(image_size, resize_mode=eval_resize_mode, mean=mean, std=std),
    )
    return DatasetBundle(train=train, val=val, test=test, num_classes=task.num_classes)


def build_fake_vtab(
    image_size: int = 32,
    num_classes: int = 10,
    train_size: int = 64,
    val_size: int = 16,
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> DatasetBundle:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train = datasets.FakeData(size=train_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)
    val = datasets.FakeData(size=val_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)
    test = datasets.FakeData(size=val_size, image_size=(3, image_size, image_size), num_classes=num_classes, transform=transform)
    return DatasetBundle(train=train, val=val, test=test, num_classes=num_classes)


def build_dataset_bundle(
    dataset: str,
    root: str | Path,
    seed: int,
    image_size: int,
    download: bool,
    strong_aug: bool,
    train_resize_mode: str = "random-resized-crop",
    eval_resize_mode: str = "resize-center-crop",
    train_size: int = 800,
    val_size: int = 200,
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    vtab_train_list: str = "train800.txt",
    vtab_val_list: str = "val200.txt",
    vtab_test_list: str = "test.txt",
) -> DatasetBundle:
    key = dataset.lower()
    if key.startswith("vtab:"):
        return build_spt_vtab(
            root=root,
            dataset=dataset,
            image_size=image_size,
            strong_aug=strong_aug,
            train_resize_mode=train_resize_mode,
            eval_resize_mode=eval_resize_mode,
            mean=mean,
            std=std,
            train_list=vtab_train_list,
            val_list=vtab_val_list,
            test_list=vtab_test_list,
        )
    try:
        task = get_vtab_task(dataset)
    except ValueError:
        task = None
    if task is not None and (Path(root) / "vtab-1k" / task.name).exists():
        return build_spt_vtab(
            root=root,
            dataset=task.name,
            image_size=image_size,
            strong_aug=strong_aug,
            train_resize_mode=train_resize_mode,
            eval_resize_mode=eval_resize_mode,
            mean=mean,
            std=std,
            train_list=vtab_train_list,
            val_list=vtab_val_list,
            test_list=vtab_test_list,
        )
    if key in {"cifar100", "cifar"}:
        return build_cifar100_vtab(
            root=root,
            seed=seed,
            image_size=image_size,
            train_size=train_size,
            val_size=val_size,
            download=download,
            strong_aug=strong_aug,
            train_resize_mode=train_resize_mode,
            eval_resize_mode=eval_resize_mode,
            mean=mean,
            std=std,
        )
    if key == "fake":
        return build_fake_vtab(image_size=image_size, mean=mean, std=std)
    raise ValueError(f"unsupported dataset: {dataset}")
