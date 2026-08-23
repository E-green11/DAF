import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "summarize_table1_vtab.py"


spec = importlib.util.spec_from_file_location("summarize_table1_vtab", MODULE_PATH)
summary_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summary_module)


def test_summary_uses_validation_loss_when_validation_accuracy_ties(tmp_path):
    metrics_dir = tmp_path / "cifar"
    metrics_dir.mkdir()
    rows = [
        {
            "epoch": 5,
            "val": {"loss": 0.1, "acc1": 100.0},
            "test": {"loss": 0.2, "acc1": 70.0},
        },
        {
            "epoch": 10,
            "val": {"loss": 0.2, "acc1": 100.0},
            "test": {"loss": 0.2, "acc1": 75.0},
        },
    ]
    (metrics_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    summary = summary_module.summarize(tmp_path, expected_epochs=200)
    cifar = next(row for row in summary if row["task"] == "cifar")

    assert cifar["best_val_epoch"] == 5
    assert cifar["test_acc1_at_best_val"] == "70.00"
