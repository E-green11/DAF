from pathlib import Path

import pytest
import torch
from torch import nn

from daf_vit import train

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 3)


def test_set_head_only_trainable_freezes_backbone_and_keeps_head_trainable():
    model = TinyClassifier()

    train.set_head_only_trainable(model)

    assert model.patch_embed.weight.requires_grad is False
    assert model.norm.weight.requires_grad is False
    assert model.head.weight.requires_grad is True
    assert model.head.bias.requires_grad is True


def test_create_model_loads_external_checkpoint_without_builtin_pretrain(monkeypatch, tmp_path):
    created = {}
    loaded = {}
    checkpoint = tmp_path / "ViT-B_16.npz"
    checkpoint.write_bytes(b"placeholder")

    def fake_create_model(model_name, pretrained, num_classes, img_size, drop_path_rate):
        created.update(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        return TinyClassifier()

    def fake_load_checkpoint(model, checkpoint_path, strict):
        loaded.update(model=model, checkpoint_path=checkpoint_path, strict=strict)

    monkeypatch.setattr(train.timm, "create_model", fake_create_model)
    monkeypatch.setattr(train, "load_checkpoint", fake_load_checkpoint)

    model = train.create_model(
        "vit_base_patch16_224",
        num_classes=100,
        pretrained=True,
        image_size=224,
        checkpoint_path=checkpoint,
        drop_path_rate=0.1,
    )

    assert isinstance(model, TinyClassifier)
    assert created["pretrained"] is False
    assert created["num_classes"] == 100
    assert created["drop_path_rate"] == 0.1
    assert loaded["checkpoint_path"] == str(checkpoint)
    assert loaded["strict"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.5,0.5,0.5", (0.5, 0.5, 0.5)),
        ("0.485, 0.456, 0.406", (0.485, 0.456, 0.406)),
    ],
)
def test_parse_float_tuple(value, expected):
    assert train.parse_float_tuple(value) == expected


def test_parse_float_tuple_rejects_wrong_length():
    with pytest.raises(ValueError):
        train.parse_float_tuple("0.5,0.5")


def test_parse_category_ratios_returns_named_ratios():
    assert train.parse_category_ratios("0.7,0.2,0.1") == {
        "matrix": 0.7,
        "vector": 0.2,
        "special": 0.1,
    }


def test_parse_category_ratios_rejects_wrong_length():
    with pytest.raises(ValueError):
        train.parse_category_ratios("0.7,0.3")


def test_can_reuse_eval_stats_when_vtab_val_and_test_lists_match():
    args = train.parse_args([])
    args.vtab_val_list = "test.txt"
    args.vtab_test_list = "test.txt"

    assert train.can_reuse_eval_stats(args)


def test_cannot_reuse_eval_stats_when_vtab_val_and_test_lists_differ():
    args = train.parse_args([])
    args.vtab_val_list = "val200.txt"
    args.vtab_test_list = "test.txt"

    assert not train.can_reuse_eval_stats(args)


def test_should_update_best_uses_validation_loss_as_accuracy_tie_breaker():
    assert train.should_update_best(90.0, 89.0, 0.5, 0.4)
    assert train.should_update_best(100.0, 100.0, 0.3, 0.4)
    assert not train.should_update_best(100.0, 100.0, 0.5, 0.4)
    assert not train.should_update_best(88.0, 89.0, 0.1, 0.9)


def test_parse_args_accepts_random_resized_crop_without_horizontal_flip():
    args = train.parse_args(["--train-resize-mode", "random-resized-crop-nohflip"])

    assert args.train_resize_mode == "random-resized-crop-nohflip"


def test_parse_args_accepts_eval_resize_mode():
    args = train.parse_args(["--eval-resize-mode", "direct-resize"])

    assert args.eval_resize_mode == "direct-resize"


def test_parse_args_accepts_drop_path_rate():
    args = train.parse_args(["--drop-path-rate", "0.1"])

    assert args.drop_path_rate == 0.1


def test_parse_args_accepts_category_ratios():
    args = train.parse_args(["--category-ratios", "0.7,0.2,0.1"])

    assert args.category_ratios == "0.7,0.2,0.1"


def test_parse_args_accepts_spt_sparse_mode():
    args = train.parse_args(["--spt-sparse", "--spt-alpha", "8", "--spt-beta", "4"])

    assert args.spt_sparse is True
    assert args.spt_alpha == 8
    assert args.spt_beta == 4


def test_table1_runner_defaults_to_val200_for_model_selection():
    script = (REPO_ROOT / "scripts" / "train_table1_vtab_daf.sh").read_text()

    assert "VTAB_TRAIN_LIST=${VTAB_TRAIN_LIST:-train800val200.txt}" in script
    assert "VTAB_VAL_LIST=${VTAB_VAL_LIST:-val200.txt}" in script
    assert "VTAB_TEST_LIST=${VTAB_TEST_LIST:-test.txt}" in script
