from daf_vit.selection import Candidate, select_budgeted


def test_select_budgeted_uses_category_budgets_and_sensitivity_order():
    candidates = [
        Candidate("m.low", "matrix", 1.0, 10),
        Candidate("m.high", "matrix", 5.0, 10),
        Candidate("v.high", "vector", 3.0, 4),
        Candidate("v.low", "vector", 0.1, 4),
        Candidate("s.high", "special", 2.0, 2),
        Candidate("s.zero", "special", 0.0, 2),
    ]

    result = select_budgeted(
        candidates,
        budget_ratio=0.5,
        category_ratios={"matrix": 0.5, "vector": 0.3, "special": 0.2},
    )

    assert result.names_by_category["matrix"] == {"m.high"}
    assert result.names_by_category["vector"] == {"v.high"}
    assert result.names_by_category["special"] == {"s.high"}
    assert "s.zero" not in result.selected_names


def test_select_budgeted_keeps_at_least_one_positive_candidate_per_category():
    candidates = [
        Candidate("m.high", "matrix", 5.0, 100),
        Candidate("v.high", "vector", 3.0, 100),
        Candidate("s.high", "special", 2.0, 100),
    ]

    result = select_budgeted(
        candidates,
        budget_ratio=0.01,
        category_ratios={"matrix": 0.34, "vector": 0.33, "special": 0.33},
    )

    assert result.selected_names == {"m.high", "v.high", "s.high"}


def test_select_budgeted_can_select_globally_by_sensitivity_density():
    candidates = [
        Candidate("m.expensive", "matrix", 100.0, 100),
        Candidate("m.dense", "matrix", 30.0, 10),
        Candidate("v.dense", "vector", 20.0, 10),
        Candidate("s.zero", "special", 0.0, 1),
    ]

    result = select_budgeted(candidates, budget_ratio=0.2, category_ratios=None, global_selection=True)

    assert result.selected_names == {"m.dense", "v.dense"}
    assert result.names_by_category["matrix"] == {"m.dense"}
    assert result.names_by_category["vector"] == {"v.dense"}
