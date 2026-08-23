from __future__ import annotations

import csv
import json
import math
import sys
import argparse
from pathlib import Path

from daf_vit.data import VTAB_TASKS


def _load_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]


def summarize(output_root: Path, expected_epochs: int = 200) -> list[dict]:
    rows = []
    for task in VTAB_TASKS:
        metrics = _load_rows(output_root / task.name / "metrics.jsonl")
        eval_rows = [row for row in metrics if not math.isnan(row["val"]["acc1"]) and not math.isnan(row["test"]["acc1"])]
        if not eval_rows:
            rows.append(
                {
                    "task": task.name,
                    "category": task.category,
                    "status": "missing",
                    "epochs": len(metrics),
                    "best_val_epoch": "",
                    "best_val_acc1": "",
                    "test_acc1_at_best_val": "",
                    "best_observed_test_epoch": "",
                    "best_observed_test_acc1": "",
                }
            )
            continue
        best_val = max(eval_rows, key=lambda row: (row["val"]["acc1"], -row["val"]["loss"], row["epoch"]))
        best_test = max(eval_rows, key=lambda row: row["test"]["acc1"])
        rows.append(
            {
                "task": task.name,
                "category": task.category,
                "status": "complete" if len(metrics) >= expected_epochs else "partial",
                "epochs": len(metrics),
                "best_val_epoch": best_val["epoch"],
                "best_val_acc1": f'{best_val["val"]["acc1"]:.2f}',
                "test_acc1_at_best_val": f'{best_val["test"]["acc1"]:.2f}',
                "best_observed_test_epoch": best_test["epoch"],
                "best_observed_test_acc1": f'{best_test["test"]["acc1"]:.2f}',
            }
        )
    return rows


def write_summary(output_root: Path, expected_epochs: int = 200) -> None:
    rows = summarize(output_root, expected_epochs=expected_epochs)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "summary.md"
    with md_path.open("w") as handle:
        handle.write("| Task | Category | Status | Epochs | Best val | Test at best val | Best observed test |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            handle.write(
                f'| {row["task"]} | {row["category"]} | {row["status"]} | {row["epochs"]} | '
                f'{row["best_val_acc1"]} | {row["test_acc1_at_best_val"]} | {row["best_observed_test_acc1"]} |\n'
            )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", nargs="?", default="outputs/table1_vtab_vit_b16_npz")
    parser.add_argument("--expected-epochs", type=int, default=200)
    args = parser.parse_args()
    write_summary(Path(args.output_root), expected_epochs=args.expected_epochs)


if __name__ == "__main__":
    main()
