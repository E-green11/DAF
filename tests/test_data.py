from daf_vit.data import make_train_val_indices


def test_make_train_val_indices_is_deterministic_and_disjoint():
    train_a, val_a = make_train_val_indices(dataset_size=100, train_size=20, val_size=10, seed=7)
    train_b, val_b = make_train_val_indices(dataset_size=100, train_size=20, val_size=10, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 20
    assert len(val_a) == 10
    assert set(train_a).isdisjoint(val_a)


def test_make_train_val_indices_rejects_oversized_split():
    try:
        make_train_val_indices(dataset_size=10, train_size=8, val_size=4, seed=0)
    except ValueError as exc:
        assert "dataset_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")

