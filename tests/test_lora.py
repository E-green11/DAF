import torch
from torch import nn

from daf_vit.lora import LoRALinear, SlicedLoRALinear


def test_lora_linear_matches_base_before_training():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    layer = LoRALinear.from_linear("proj", base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(5, 4)

    torch.testing.assert_close(layer(x), base(x))


def test_lora_adapter_can_be_activated_frozen_and_merged():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    layer = LoRALinear.from_linear("proj", base, rank=2, alpha=2.0, dropout=0.0)
    layer.ensure_adapter("stage0", trainable=True)
    layer.set_active("stage0")

    assert all(param.requires_grad for param in layer.adapters["stage0"].parameters())

    with torch.no_grad():
        layer.adapters["stage0"].down.weight.fill_(0.25)
        layer.adapters["stage0"].up.weight.fill_(0.5)

    x = torch.randn(7, 4)
    adapted = layer(x)
    assert not torch.allclose(adapted, layer.base(x))

    layer.freeze_adapter("stage0")
    assert not any(param.requires_grad for param in layer.adapters["stage0"].parameters())

    merged = layer.merge_all()
    torch.testing.assert_close(merged(x), adapted, atol=1e-6, rtol=1e-6)


def test_lora_linear_preserves_base_dtype_for_late_wrapping():
    base = nn.Linear(4, 3).to(dtype=torch.float64)
    layer = LoRALinear.from_linear("proj", base, rank=2, alpha=2.0, dropout=0.0)
    layer.ensure_adapter("stage0", trainable=True)

    assert layer.base.weight.dtype == torch.float64
    assert layer.adapters["stage0"].down.weight.dtype == torch.float64


def test_sliced_lora_linear_matches_base_before_training():
    torch.manual_seed(0)
    base = nn.Linear(4, 12)
    layer = SlicedLoRALinear.from_qkv_linear("qkv", base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(5, 4)

    torch.testing.assert_close(layer(x), base(x))


def test_sliced_lora_linear_trains_only_selected_slice_adapter():
    torch.manual_seed(0)
    base = nn.Linear(4, 12)
    layer = SlicedLoRALinear.from_qkv_linear("qkv", base, rank=2, alpha=4.0, dropout=0.0)

    layer.ensure_adapter("stage0", part="q", trainable=True)

    assert "stage0:q" in layer.adapters
    assert all(param.requires_grad for param in layer.adapters["stage0:q"].parameters())
    assert "stage0:k" not in layer.adapters
    assert "stage0:v" not in layer.adapters
