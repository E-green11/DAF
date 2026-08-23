import torch
from torch import nn

from daf_vit.lora import LoRALinear


def test_lora_linear_sparse_delta_updates_selected_weight_and_bias_entries():
    base = nn.Linear(3, 2)
    with torch.no_grad():
        base.weight.zero_()
        base.bias.zero_()
    layer = LoRALinear.from_linear("linear", base, rank=1, alpha=1, dropout=0)

    layer.ensure_sparse_delta(
        "stage0",
        weight_indices=torch.tensor([[0, 1], [1, 2]]),
        bias_indices=torch.tensor([1]),
        trainable=True,
    )
    with torch.no_grad():
        layer.sparse_weight_deltas["stage0_weight"].copy_(torch.tensor([2.0, 3.0]))
        layer.sparse_bias_deltas["stage0_bias"].copy_(torch.tensor([5.0]))

    output = layer(torch.tensor([[10.0, 20.0, 30.0]]))

    assert torch.allclose(output, torch.tensor([[40.0, 95.0]]))
    assert layer.sparse_weight_deltas["stage0_weight"].requires_grad
    assert layer.sparse_bias_deltas["stage0_bias"].requires_grad
