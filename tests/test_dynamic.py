import torch
from torch import nn

from daf_vit.dynamic import SensitivityAnalyzer, set_trainable_state, wrap_vit_lora
from daf_vit.lora import LoRALinear, SlicedLoRALinear


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(4, 12)
        self.attn.proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 4))
        self.blocks = nn.ModuleList([TinyBlock()])


def test_wrap_vit_lora_replaces_target_linears_only():
    model = TinyModel()

    wrapped = wrap_vit_lora(model, rank=2, alpha=4.0, dropout=0.0)

    assert isinstance(model.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(model.blocks[0].attn.proj, LoRALinear)
    assert isinstance(model.blocks[0].mlp.fc1, LoRALinear)
    assert isinstance(model.blocks[0].head, nn.Linear)
    assert set(wrapped) == {
        "blocks.0.attn.qkv",
        "blocks.0.attn.proj",
        "blocks.0.mlp.fc1",
    }


def test_set_trainable_state_refocuses_lora_and_vector_parameters():
    model = TinyModel()
    wrap_vit_lora(model, rank=2, alpha=4.0, dropout=0.0)

    set_trainable_state(
        model,
        active_matrices={"blocks.0.attn.qkv"},
        active_vectors={"blocks.0.norm.weight"},
        active_specials={"cls_token"},
        adapter_id="stage0",
    )

    assert any(param.requires_grad for param in model.blocks[0].attn.qkv.adapters["stage0"].parameters())
    assert "stage0" not in model.blocks[0].attn.proj.adapters
    assert model.blocks[0].norm.weight.requires_grad
    assert model.blocks[0].norm.bias.requires_grad is False
    assert model.cls_token.requires_grad
    assert model.pos_embed.requires_grad is False
    assert model.blocks[0].head.weight.requires_grad
    assert model.blocks[0].head.bias.requires_grad


def test_wrap_vit_lora_can_split_fused_qkv_candidates():
    model = TinyModel()

    wrapped = wrap_vit_lora(model, rank=2, alpha=4.0, dropout=0.0, split_qkv=True)

    assert isinstance(model.blocks[0].attn.qkv, SlicedLoRALinear)
    assert set(wrapped) == {
        "blocks.0.attn.qkv.q",
        "blocks.0.attn.qkv.k",
        "blocks.0.attn.qkv.v",
        "blocks.0.attn.proj",
        "blocks.0.mlp.fc1",
    }


def test_set_trainable_state_refocuses_split_qkv_adapter():
    model = TinyModel()
    wrap_vit_lora(model, rank=2, alpha=4.0, dropout=0.0, split_qkv=True)

    set_trainable_state(
        model,
        active_matrices={"blocks.0.attn.qkv.q"},
        active_vectors=set(),
        active_specials=set(),
        adapter_id="stage0",
    )

    qkv = model.blocks[0].attn.qkv
    assert "stage0:q" in qkv.adapters
    assert "stage0:k" not in qkv.adapters
    assert "stage0:v" not in qkv.adapters


def test_sensitivity_analyzer_supports_grad_square_scores():
    param = nn.Parameter(torch.tensor([2.0, -3.0]))
    param.grad = torch.tensor([0.5, -4.0])

    analyzer = SensitivityAnalyzer(score_mode="grad_square")

    assert analyzer._parameter_score(param) == torch.tensor(16.25).item()


def test_sensitivity_analyzer_supports_grad_weight_abs_scores():
    param = nn.Parameter(torch.tensor([2.0, -3.0]))
    param.grad = torch.tensor([0.5, -4.0])

    analyzer = SensitivityAnalyzer(score_mode="grad_weight_abs")

    assert analyzer._parameter_score(param) == torch.tensor(13.0).item()
