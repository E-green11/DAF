from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(x))) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.up.weight @ self.down.weight) * self.scaling


class LoRALinear(nn.Module):
    """A frozen base linear layer with dynamically managed LoRA adapters."""

    def __init__(self, name: str, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.name = name
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout
        self.base = base
        self.adapters = nn.ModuleDict()
        self.sparse_weight_deltas = nn.ParameterDict()
        self.sparse_bias_deltas = nn.ParameterDict()
        self._sparse_weight_indices: dict[str, torch.Tensor] = {}
        self._sparse_bias_indices: dict[str, torch.Tensor] = {}
        self.active_adapter: str | None = None
        for param in self.base.parameters():
            param.requires_grad = False

    @classmethod
    def from_linear(cls, name: str, linear: nn.Linear, rank: int, alpha: float, dropout: float) -> "LoRALinear":
        base = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        base.to(device=linear.weight.device, dtype=linear.weight.dtype)
        base.load_state_dict(linear.state_dict())
        return cls(name=name, base=base, rank=rank, alpha=alpha, dropout=dropout)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def ensure_adapter(self, adapter_id: str, trainable: bool) -> None:
        if adapter_id not in self.adapters:
            adapter = LoRAAdapter(
                self.in_features,
                self.out_features,
                self.rank,
                self.alpha,
                self.dropout_p,
            )
            adapter.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.adapters[adapter_id] = adapter
        for param in self.adapters[adapter_id].parameters():
            param.requires_grad = trainable

    def set_active(self, adapter_id: str | None) -> None:
        if adapter_id is not None and adapter_id not in self.adapters:
            raise KeyError(f"unknown LoRA adapter: {adapter_id}")
        self.active_adapter = adapter_id

    def freeze_adapter(self, adapter_id: str) -> None:
        if adapter_id not in self.adapters:
            return
        for param in self.adapters[adapter_id].parameters():
            param.requires_grad = False

    def freeze_all_adapters(self) -> None:
        for adapter_id in self.adapters:
            self.freeze_adapter(adapter_id)
        for param in self.sparse_weight_deltas.values():
            param.requires_grad = False
        for param in self.sparse_bias_deltas.values():
            param.requires_grad = False

    def ensure_sparse_delta(
        self,
        adapter_id: str,
        weight_indices: torch.Tensor | None = None,
        bias_indices: torch.Tensor | None = None,
        trainable: bool = True,
    ) -> None:
        if weight_indices is not None and weight_indices.numel() > 0:
            key = f"{adapter_id}_weight"
            if key not in self.sparse_weight_deltas:
                indices = weight_indices.to(device=self.base.weight.device, dtype=torch.long)
                if indices.ndim != 2 or indices.shape[0] != 2:
                    raise ValueError("weight_indices must have shape (2, n)")
                self.sparse_weight_deltas[key] = nn.Parameter(
                    torch.zeros(indices.shape[1], device=self.base.weight.device, dtype=self.base.weight.dtype)
                )
                self._sparse_weight_indices[key] = indices
            self.sparse_weight_deltas[key].requires_grad = trainable
        if bias_indices is not None and bias_indices.numel() > 0:
            key = f"{adapter_id}_bias"
            if key not in self.sparse_bias_deltas:
                indices = bias_indices.flatten().to(device=self.base.weight.device, dtype=torch.long)
                self.sparse_bias_deltas[key] = nn.Parameter(
                    torch.zeros(indices.shape[0], device=self.base.weight.device, dtype=self.base.weight.dtype)
                )
                self._sparse_bias_indices[key] = indices
            self.sparse_bias_deltas[key].requires_grad = trainable

    def _sparse_update(self, x: torch.Tensor) -> torch.Tensor | None:
        update = None
        flat_x = x.reshape(-1, self.in_features)
        for key, delta in self.sparse_weight_deltas.items():
            indices = self._sparse_weight_indices[key].to(device=x.device)
            rows = indices[0]
            cols = indices[1]
            flat_update = torch.zeros(flat_x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
            flat_update.index_add_(1, rows, flat_x[:, cols] * delta.to(dtype=x.dtype))
            shaped = flat_update.reshape(*x.shape[:-1], self.out_features)
            update = shaped if update is None else update + shaped
        for key, delta in self.sparse_bias_deltas.items():
            indices = self._sparse_bias_indices[key].to(device=x.device)
            bias_update = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
            bias_update.index_add_(0, indices, delta.to(dtype=x.dtype))
            shaped = bias_update.view(*([1] * (x.ndim - 1)), self.out_features)
            update = shaped if update is None else update + shaped
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        sparse_update = self._sparse_update(x)
        if sparse_update is not None:
            y = y + sparse_update
        for adapter in self.adapters.values():
            y = y + adapter(x)
        return y

    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base.weight.detach().clone()
        for adapter in self.adapters.values():
            weight = weight + adapter.delta_weight().detach().to(weight.device, weight.dtype)
        bias = None if self.base.bias is None else self.base.bias.detach().clone()
        return weight, bias

    def merge_all(self) -> nn.Linear:
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        weight, bias = self.merged_weight_bias()
        with torch.no_grad():
            merged.weight.copy_(weight)
            if bias is not None:
                merged.bias.copy_(bias)
        merged.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        return merged

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, adapters={list(self.adapters.keys())}"
        )


class SlicedLoRALinear(nn.Module):
    """A frozen fused qkv linear layer with independent LoRA adapters per q/k/v slice."""

    def __init__(
        self,
        name: str,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        parts: dict[str, tuple[int, int]],
    ) -> None:
        super().__init__()
        self.name = name
        self.rank = rank
        self.alpha = alpha
        self.dropout_p = dropout
        self.base = base
        self.parts = parts
        self.adapters = nn.ModuleDict()
        self.sparse_weight_deltas = nn.ParameterDict()
        self.sparse_bias_deltas = nn.ParameterDict()
        self._sparse_weight_indices: dict[str, torch.Tensor] = {}
        self._sparse_bias_indices: dict[str, torch.Tensor] = {}
        for param in self.base.parameters():
            param.requires_grad = False

    @classmethod
    def from_qkv_linear(cls, name: str, linear: nn.Linear, rank: int, alpha: float, dropout: float) -> "SlicedLoRALinear":
        if linear.out_features % 3 != 0:
            raise ValueError(f"qkv linear out_features must be divisible by 3, got {linear.out_features}")
        base = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        base.to(device=linear.weight.device, dtype=linear.weight.dtype)
        base.load_state_dict(linear.state_dict())
        width = linear.out_features // 3
        return cls(
            name=name,
            base=base,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            parts={"q": (0, width), "k": (width, 2 * width), "v": (2 * width, 3 * width)},
        )

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def part_out_features(self, part: str) -> int:
        start, end = self.parts[part]
        return end - start

    def adapter_key(self, adapter_id: str, part: str) -> str:
        return f"{adapter_id}:{part}"

    def ensure_adapter(self, adapter_id: str, part: str, trainable: bool) -> None:
        if part not in self.parts:
            raise KeyError(f"unknown slice part: {part}")
        key = self.adapter_key(adapter_id, part)
        if key not in self.adapters:
            adapter = LoRAAdapter(
                self.in_features,
                self.part_out_features(part),
                self.rank,
                self.alpha,
                self.dropout_p,
            )
            adapter.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            self.adapters[key] = adapter
        for param in self.adapters[key].parameters():
            param.requires_grad = trainable

    def freeze_adapter(self, adapter_id: str, part: str | None = None) -> None:
        prefix = f"{adapter_id}:" if part is None else self.adapter_key(adapter_id, part)
        for key, adapter in self.adapters.items():
            if key == prefix or key.startswith(prefix):
                for param in adapter.parameters():
                    param.requires_grad = False

    def freeze_all_adapters(self) -> None:
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = False
        for param in self.sparse_weight_deltas.values():
            param.requires_grad = False
        for param in self.sparse_bias_deltas.values():
            param.requires_grad = False

    def ensure_sparse_delta(
        self,
        adapter_id: str,
        weight_indices: torch.Tensor | None = None,
        bias_indices: torch.Tensor | None = None,
        trainable: bool = True,
    ) -> None:
        if weight_indices is not None and weight_indices.numel() > 0:
            key = f"{adapter_id}_weight"
            if key not in self.sparse_weight_deltas:
                indices = weight_indices.to(device=self.base.weight.device, dtype=torch.long)
                if indices.ndim != 2 or indices.shape[0] != 2:
                    raise ValueError("weight_indices must have shape (2, n)")
                self.sparse_weight_deltas[key] = nn.Parameter(
                    torch.zeros(indices.shape[1], device=self.base.weight.device, dtype=self.base.weight.dtype)
                )
                self._sparse_weight_indices[key] = indices
            self.sparse_weight_deltas[key].requires_grad = trainable
        if bias_indices is not None and bias_indices.numel() > 0:
            key = f"{adapter_id}_bias"
            if key not in self.sparse_bias_deltas:
                indices = bias_indices.flatten().to(device=self.base.weight.device, dtype=torch.long)
                self.sparse_bias_deltas[key] = nn.Parameter(
                    torch.zeros(indices.shape[0], device=self.base.weight.device, dtype=self.base.weight.dtype)
                )
                self._sparse_bias_indices[key] = indices
            self.sparse_bias_deltas[key].requires_grad = trainable

    def _sparse_update(self, x: torch.Tensor) -> torch.Tensor | None:
        update = None
        flat_x = x.reshape(-1, self.in_features)
        for key, delta in self.sparse_weight_deltas.items():
            indices = self._sparse_weight_indices[key].to(device=x.device)
            rows = indices[0]
            cols = indices[1]
            flat_update = torch.zeros(flat_x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
            flat_update.index_add_(1, rows, flat_x[:, cols] * delta.to(dtype=x.dtype))
            shaped = flat_update.reshape(*x.shape[:-1], self.out_features)
            update = shaped if update is None else update + shaped
        for key, delta in self.sparse_bias_deltas.items():
            indices = self._sparse_bias_indices[key].to(device=x.device)
            bias_update = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
            bias_update.index_add_(0, indices, delta.to(dtype=x.dtype))
            shaped = bias_update.view(*([1] * (x.ndim - 1)), self.out_features)
            update = shaped if update is None else update + shaped
        return update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        sparse_update = self._sparse_update(x)
        if sparse_update is not None:
            y = y + sparse_update
        chunks = []
        for part, (start, end) in self.parts.items():
            chunk = y[..., start:end]
            for key, adapter in self.adapters.items():
                _, adapter_part = key.split(":", 1)
                if adapter_part != part:
                    continue
                chunk = chunk + adapter(x)
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, parts={list(self.parts)}, adapters={list(self.adapters.keys())}"
        )


def lora_delta_linear(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, scaling: float) -> torch.Tensor:
    return F.linear(F.linear(x, down), up) * scaling
