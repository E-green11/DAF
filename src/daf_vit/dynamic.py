from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .lora import LoRALinear, SlicedLoRALinear
from .selection import Candidate, SelectionResult, select_budgeted

DEFAULT_TARGET_SUBSTRINGS = (
    ".attn.qkv",
    ".attn.proj",
    ".mlp.fc1",
    ".mlp.fc2",
)
DEFAULT_CATEGORY_RATIOS = {"matrix": 0.3, "vector": 0.4, "special": 0.3}
SPT_SKIP_SUBSTRINGS = ("head", "cls_token", "patch_embed", "pos_embed")


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _matches_target(name: str, target_substrings: Iterable[str]) -> bool:
    dotted = f".{name}"
    return any(target in dotted for target in target_substrings)


def wrap_vit_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_substrings: tuple[str, ...] = DEFAULT_TARGET_SUBSTRINGS,
    split_qkv: bool = False,
) -> dict[str, LoRALinear | SlicedLoRALinear]:
    wrapped: dict[str, LoRALinear | SlicedLoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, target_substrings):
            continue
        parent, child_name = _get_parent(model, name)
        if split_qkv and name.endswith("attn.qkv"):
            lora = SlicedLoRALinear.from_qkv_linear(name, module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora)
            for part in lora.parts:
                wrapped[f"{name}.{part}"] = lora
            continue
        lora = LoRALinear.from_linear(name, module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora)
        wrapped[name] = lora
    return wrapped


def iter_lora_layers(model: nn.Module) -> Iterable[tuple[str, LoRALinear | SlicedLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, SlicedLoRALinear)):
            yield name, module


def set_trainable_state(
    model: nn.Module,
    active_matrices: set[str],
    active_vectors: set[str],
    active_specials: set[str],
    adapter_id: str,
    active_sparse: dict[str, torch.Tensor] | None = None,
) -> None:
    active_sparse = active_sparse or {}
    for param in model.parameters():
        param.requires_grad = False

    for name, layer in iter_lora_layers(model):
        layer.freeze_all_adapters()
        layer.ensure_sparse_delta(
            adapter_id,
            weight_indices=active_sparse.get(f"{name}.base.weight"),
            bias_indices=active_sparse.get(f"{name}.base.bias"),
            trainable=True,
        )
        if isinstance(layer, SlicedLoRALinear):
            for part in layer.parts:
                if f"{name}.{part}" in active_matrices:
                    layer.ensure_adapter(adapter_id, part=part, trainable=True)
        elif name in active_matrices:
            layer.ensure_adapter(adapter_id, trainable=True)
            layer.set_active(adapter_id)

    for name, param in model.named_parameters():
        if name in active_vectors or name in active_specials:
            param.requires_grad = True
        if ".head." in f".{name}" or name.startswith("head."):
            param.requires_grad = True


def lora_cost(layer: LoRALinear | SlicedLoRALinear, rank: int, part: str | None = None) -> int:
    out_features = layer.part_out_features(part) if isinstance(layer, SlicedLoRALinear) and part else layer.out_features
    return (layer.in_features + out_features) * rank


def candidate_category(name: str, param: nn.Parameter) -> str | None:
    if "head" in name:
        return None
    if "cls_token" in name or "pos_embed" in name:
        return "special"
    if param.ndim == 1 and ("norm" in name or name.endswith("bias")):
        return "vector"
    return None


@dataclass
class SensitivityAnalyzer:
    batch_num: int = 16
    rank: int = 8
    score_mode: str = "grad_weight_abs"

    def analyze(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        device: torch.device,
    ) -> list[Candidate]:
        scores = self.analyze_scores(model, dataloader, criterion, device)
        return self._scores_to_candidates(model, scores)

    def analyze_scores(
        self,
        model: nn.Module,
        dataloader,
        criterion: nn.Module,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        previous_trainable = {name: param.requires_grad for name, param in model.named_parameters()}
        try:
            self._enable_backbone_sensitivity_grads(model)
            scores: dict[str, torch.Tensor] = {}
            model.train()
            for batch_index, (samples, targets) in enumerate(dataloader):
                if batch_index >= self.batch_num:
                    break
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                loss = criterion(model(samples), targets)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    score = self._parameter_score_tensor(param).cpu()
                    if name in scores:
                        scores[name] = scores[name] + score
                    else:
                        scores[name] = score
            return scores
        finally:
            for name, param in model.named_parameters():
                param.requires_grad = previous_trainable[name]
            model.zero_grad(set_to_none=True)

    def _parameter_score(self, param: nn.Parameter) -> float:
        if param.grad is None:
            return 0.0
        grad = param.grad.detach()
        if self.score_mode == "grad_square":
            return torch.sum(grad.square()).item()
        if self.score_mode == "grad_weight_abs":
            return torch.sum(torch.abs(grad * param.detach())).item()
        raise ValueError(f"unsupported sensitivity score mode: {self.score_mode}")

    def _parameter_score_tensor(self, param: nn.Parameter) -> torch.Tensor:
        if param.grad is None:
            return torch.zeros_like(param.detach(), dtype=torch.float32)
        grad = param.grad.detach().float()
        if self.score_mode == "grad_square":
            return grad.square()
        if self.score_mode == "grad_weight_abs":
            return torch.abs(grad * param.detach().float())
        raise ValueError(f"unsupported sensitivity score mode: {self.score_mode}")

    def _enable_backbone_sensitivity_grads(self, model: nn.Module) -> None:
        for _, layer in iter_lora_layers(model):
            layer.freeze_all_adapters()
            for param in layer.base.parameters():
                param.requires_grad = True
        for name, param in model.named_parameters():
            category = candidate_category(name, param)
            if category is not None:
                param.requires_grad = True

    def _score_sum(self, scores: dict[str, torch.Tensor], name: str) -> float:
        score = scores.get(name)
        if score is None:
            return 0.0
        return score.sum().item()

    def _scores_to_candidates(self, model: nn.Module, scores: dict[str, torch.Tensor]) -> list[Candidate]:
        candidates: list[Candidate] = []
        for name, layer in iter_lora_layers(model):
            weight_name = f"{name}.base.weight"
            if isinstance(layer, SlicedLoRALinear):
                weight_scores = scores.get(weight_name)
                for part, (start, end) in layer.parts.items():
                    if weight_scores is None:
                        score = 0.0
                    else:
                        score = weight_scores[start:end].sum().item()
                    candidates.append(
                        Candidate(
                            name=f"{name}.{part}",
                            category="matrix",
                            sensitivity=score,
                            cost=lora_cost(layer, self.rank, part=part),
                        )
                    )
            else:
                candidates.append(
                    Candidate(
                        name=name,
                        category="matrix",
                        sensitivity=self._score_sum(scores, weight_name),
                        cost=lora_cost(layer, self.rank),
                    )
                )
        for name, param in model.named_parameters():
            category = candidate_category(name, param)
            if category is None:
                continue
            candidates.append(Candidate(name=name, category=category, sensitivity=self._score_sum(scores, name), cost=param.numel()))
        return candidates


def _append_sparse_index(target: dict[str, torch.Tensor], name: str, indices: torch.Tensor) -> int:
    if indices.numel() == 0:
        return 0
    indices = indices.cpu()
    if name in target:
        target[name] = torch.cat([target[name], indices], dim=1 if indices.ndim == 2 else 0)
    else:
        target[name] = indices
    return indices.shape[1] if indices.ndim == 2 else indices.numel()


def _select_sparse_or_structured_weight(
    name: str,
    mask: torch.Tensor,
    layer: LoRALinear | SlicedLoRALinear,
    rank: int,
    alpha: float,
    selected: list[Candidate],
    names_by_category: dict[str, set[str]],
    sparse_indices: dict[str, torch.Tensor],
    part: str | None = None,
    row_offset: int = 0,
) -> int:
    count = int(mask.sum().item())
    if count == 0:
        return 0
    cost = lora_cost(layer, rank, part=part)
    score = float(count)
    matrix_name = f"{layer.name}.{part}" if part else layer.name
    if count >= cost / alpha:
        candidate = Candidate(matrix_name, "matrix", score, cost)
        selected.append(candidate)
        names_by_category.setdefault("matrix", set()).add(candidate.name)
        return cost
    indices = mask.nonzero(as_tuple=False).t()
    if row_offset:
        indices[0] += row_offset
    sparse_cost = _append_sparse_index(sparse_indices, name, indices)
    candidate = Candidate(f"sparse:{name}:{len(names_by_category.get('unstructured', set()))}", "unstructured", score, sparse_cost)
    selected.append(candidate)
    names_by_category.setdefault("unstructured", set()).add(candidate.name)
    return sparse_cost


def select_spt_like(
    model: nn.Module,
    scores: dict[str, torch.Tensor],
    budget_million: float,
    rank: int,
    alpha: float = 10.0,
    beta: float = 5.0,
) -> SelectionResult:
    if budget_million <= 0:
        raise ValueError("budget_million must be positive")

    layer_by_name = {name: layer for name, layer in iter_lora_layers(model)}
    score_items = [
        (name, score.float())
        for name, score in scores.items()
        if not any(skip in name for skip in SPT_SKIP_SUBSTRINGS)
    ]
    if not score_items:
        raise RuntimeError("no sensitivity scores available for SPT-like sparse selection")

    flat_scores = torch.cat([score.flatten() for _, score in score_items])
    target = max(1, int(round(budget_million * 1_000_000)))
    max_probe = min(flat_scores.numel(), max(target, int(round(budget_million * 2_000_000))))
    _, top_indices = flat_scores.topk(max_probe)
    flat_ranks = torch.full((flat_scores.numel(),), max_probe + 1, dtype=torch.int32)
    flat_ranks[top_indices.cpu()] = torch.arange(max_probe, dtype=torch.int32)
    named_params = dict(model.named_parameters())

    def build_for_topk(topk: int) -> SelectionResult:
        selected: list[Candidate] = []
        names_by_category: dict[str, set[str]] = {}
        sparse_indices: dict[str, torch.Tensor] = {}
        total_cost = 0
        offset = 0
        for name, score in score_items:
            numel = score.numel()
            mask = (flat_ranks[offset : offset + numel] < topk).reshape(score.shape)
            offset += numel
            count = int(mask.sum().item())
            if count == 0:
                continue

            if name.endswith(".base.weight"):
                layer_name = name[: -len(".base.weight")]
                layer = layer_by_name.get(layer_name)
                if layer is None:
                    continue
                if isinstance(layer, SlicedLoRALinear):
                    for part, (start, end) in layer.parts.items():
                        part_mask = mask[start:end]
                        total_cost += _select_sparse_or_structured_weight(
                            name,
                            part_mask,
                            layer,
                            rank,
                            alpha,
                            selected,
                            names_by_category,
                            sparse_indices,
                            part=part,
                            row_offset=start,
                        )
                else:
                    total_cost += _select_sparse_or_structured_weight(
                        name,
                        mask,
                        layer,
                        rank,
                        alpha,
                        selected,
                        names_by_category,
                        sparse_indices,
                    )
                continue

            if name.endswith(".base.bias"):
                layer_name = name[: -len(".base.bias")]
                layer = layer_by_name.get(layer_name)
                if layer is None:
                    continue
                if count >= mask.numel() / beta:
                    candidate = Candidate(name, "vector", float(score[mask].sum().item()), mask.numel())
                    selected.append(candidate)
                    names_by_category.setdefault("vector", set()).add(name)
                    total_cost += candidate.cost
                else:
                    indices = mask.nonzero(as_tuple=False).flatten()
                    sparse_cost = _append_sparse_index(sparse_indices, name, indices)
                    candidate = Candidate(f"sparse:{name}", "unstructured", float(score[mask].sum().item()), sparse_cost)
                    selected.append(candidate)
                    names_by_category.setdefault("unstructured", set()).add(candidate.name)
                    total_cost += sparse_cost
                continue

            param = named_params.get(name)
            category = candidate_category(name, param) if param is not None else None
            if category == "vector" and count >= mask.numel() / beta:
                candidate = Candidate(name, "vector", float(score[mask].sum().item()), mask.numel())
                selected.append(candidate)
                names_by_category.setdefault("vector", set()).add(name)
                total_cost += candidate.cost

        for name, param in named_params.items():
            if name == "cls_token":
                candidate = Candidate(name, "special", 1.0, param.numel())
                selected.append(candidate)
                names_by_category.setdefault("special", set()).add(name)
                total_cost += candidate.cost
                break

        return SelectionResult(
            selected=tuple(selected),
            total_cost=total_cost,
            budget=target,
            names_by_category=names_by_category,
            sparse_indices=sparse_indices,
        )

    step = max(1, int(round(0.02 * 1_000_000)))
    probe_topks = sorted({min(max_probe, step * i) for i in range(1, max(2, max_probe // step + 1))} | {target})
    best = min((build_for_topk(topk) for topk in probe_topks), key=lambda item: abs(item.total_cost - target))
    return best


@dataclass
class DynamicDAFManager:
    budget_ratio: float = 0.2
    update_interval: int = 10
    category_ratios: dict[str, float] | None = None
    selection_mode: str = "category"
    stage: int = 0
    current: SelectionResult | None = None

    def initialize_selection(self, model: nn.Module, selection: SelectionResult) -> SelectionResult:
        self.stage = 0
        self.current = selection
        self.apply(model, self.current)
        return self.current

    def initialize(self, model: nn.Module, candidates: list[Candidate]) -> SelectionResult:
        self.stage = 0
        self.current = select_budgeted(
            candidates,
            self.budget_ratio,
            self.category_ratios or DEFAULT_CATEGORY_RATIOS,
            global_selection=self.selection_mode == "global",
        )
        self.apply(model, self.current)
        return self.current

    def maybe_update(self, epoch: int, model: nn.Module, candidates: list[Candidate]) -> SelectionResult | None:
        if epoch <= 0 or epoch % self.update_interval != 0:
            return None
        self.stage += 1
        self.current = select_budgeted(
            candidates,
            self.budget_ratio,
            self.category_ratios or DEFAULT_CATEGORY_RATIOS,
            global_selection=self.selection_mode == "global",
        )
        self.apply(model, self.current)
        return self.current

    def maybe_update_selection(self, epoch: int, model: nn.Module, selection: SelectionResult) -> SelectionResult | None:
        if epoch <= 0 or epoch % self.update_interval != 0:
            return None
        self.stage += 1
        self.current = selection
        self.apply(model, self.current)
        return self.current

    def apply(self, model: nn.Module, selection: SelectionResult) -> None:
        names = selection.names_by_category
        set_trainable_state(
            model,
            active_matrices=names.get("matrix", set()),
            active_vectors=names.get("vector", set()),
            active_specials=names.get("special", set()),
            adapter_id=f"stage{self.stage}",
            active_sparse=selection.sparse_indices,
        )
