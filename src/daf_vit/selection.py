from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class Candidate:
    name: str
    category: str
    sensitivity: float
    cost: int


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...]
    total_cost: int
    budget: int
    names_by_category: dict[str, set[str]]
    sparse_indices: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def selected_names(self) -> set[str]:
        return {candidate.name for candidate in self.selected}


def select_budgeted(
    candidates: Sequence[Candidate],
    budget_ratio: float,
    category_ratios: Mapping[str, float] | None = None,
    global_selection: bool = False,
) -> SelectionResult:
    if not 0 < budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if category_ratios is None and not global_selection:
        category_ratios = {"matrix": 0.3, "vector": 0.4, "special": 0.3}

    positive = [candidate for candidate in candidates if candidate.sensitivity > 0 and candidate.cost > 0]
    total_possible = sum(candidate.cost for candidate in positive)
    total_budget = max(1, int(total_possible * budget_ratio))

    if global_selection:
        selected: list[Candidate] = []
        spent = 0
        for candidate in sorted(positive, key=lambda item: item.sensitivity / item.cost, reverse=True):
            if spent + candidate.cost <= total_budget:
                selected.append(candidate)
                spent += candidate.cost
        if not selected and positive:
            selected = [max(positive, key=lambda item: item.sensitivity / item.cost)]
        names_by_category: dict[str, set[str]] = {}
        for candidate in selected:
            names_by_category.setdefault(candidate.category, set()).add(candidate.name)
        return SelectionResult(
            selected=tuple(selected),
            total_cost=sum(candidate.cost for candidate in selected),
            budget=total_budget,
            names_by_category=names_by_category,
        )

    selected: list[Candidate] = []
    names_by_category = {category: set() for category in category_ratios}

    for category, ratio in category_ratios.items():
        category_candidates = sorted(
            (candidate for candidate in positive if candidate.category == category),
            key=lambda item: item.sensitivity,
            reverse=True,
        )
        if not category_candidates:
            continue
        category_budget = max(1, int(total_budget * ratio))
        spent = 0
        category_selected: list[Candidate] = []
        for candidate in category_candidates:
            if spent + candidate.cost <= category_budget:
                category_selected.append(candidate)
                spent += candidate.cost
        if not category_selected:
            category_selected.append(category_candidates[0])
            spent = category_candidates[0].cost
        selected.extend(category_selected)
        names_by_category[category] = {candidate.name for candidate in category_selected}

    return SelectionResult(
        selected=tuple(selected),
        total_cost=sum(candidate.cost for candidate in selected),
        budget=total_budget,
        names_by_category=names_by_category,
    )
