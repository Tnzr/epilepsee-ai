#!/usr/bin/env python
"""Summarize sampling-stride sweep outputs into CSV/JSON.

Expected structure (created by Make sweep targets):
  models/sampling_sweeps/
    stride_0p5_20260311_150000/
      results.json
      run_metadata.json
      ...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List


def _safe_get(container: Dict[str, Any], *keys: str, default=None):
    current = container
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _parse_stride_from_name(name: str):
    match = re.search(r"stride_([0-9p]+)", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def collect_rows(sweep_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted([path for path in sweep_dir.iterdir() if path.is_dir()]):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue

        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        metadata_path = run_dir / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        stride_seconds = metadata.get("stride_seconds")
        if stride_seconds is None:
            stride_seconds = _parse_stride_from_name(run_dir.name)

        test_metrics = results.get("test_metrics", {})
        threshold_sweep = results.get("threshold_sweep", {})

        rows.append(
            {
                "run_dir": run_dir.name,
                "stride_seconds": stride_seconds,
                "mode": metadata.get("mode", "unknown"),
                "seed": metadata.get("seed"),
                "threshold": _safe_get(threshold_sweep, "threshold"),
                "objective": _safe_get(threshold_sweep, "objective"),
                "accuracy": _safe_get(test_metrics, "accuracy"),
                "auroc": _safe_get(test_metrics, "auroc"),
                "f1_score": _safe_get(test_metrics, "f1_score"),
                "sensitivity": _safe_get(test_metrics, "sensitivity"),
                "specificity": _safe_get(test_metrics, "specificity"),
                "precision": _safe_get(test_metrics, "precision"),
                "fpr_per_8hours": _safe_get(test_metrics, "fpr_per_8hours"),
                "mae": _safe_get(test_metrics, "mae"),
                "rmse": _safe_get(test_metrics, "rmse"),
            }
        )

    rows.sort(key=lambda row: (float("inf") if row["stride_seconds"] is None else row["stride_seconds"], row["run_dir"]))
    return rows


def write_outputs(rows: List[Dict[str, Any]], sweep_dir: Path) -> None:
    summary_json = sweep_dir / "sampling_sweep_summary.json"
    summary_csv = sweep_dir / "sampling_sweep_summary.csv"

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with summary_csv.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_csv.write_text("", encoding="utf-8")

    print(f"Found {len(rows)} sweep runs")
    print(f"Wrote JSON: {summary_json}")
    print(f"Wrote CSV : {summary_csv}")

    if rows:
        print("\nStride summary (top rows):")
        header = "stride_s  sens    spec    auroc   f1      fpr/8h   mae     run_dir"
        print(header)
        print("-" * len(header))
        for row in rows[:15]:
            stride = row["stride_seconds"]
            sensitivity = row["sensitivity"]
            specificity = row["specificity"]
            auroc = row["auroc"]
            f1_score = row["f1_score"]
            fpr_8h = row["fpr_per_8hours"]
            mae = row["mae"]
            print(
                f"{(stride if stride is not None else 'NA'):>7}  "
                f"{(sensitivity if sensitivity is not None else float('nan')):>6.3f}  "
                f"{(specificity if specificity is not None else float('nan')):>6.3f}  "
                f"{(auroc if auroc is not None else float('nan')):>6.3f}  "
                f"{(f1_score if f1_score is not None else float('nan')):>6.3f}  "
                f"{(fpr_8h if fpr_8h is not None else float('nan')):>7.1f}  "
                f"{(mae if mae is not None else float('nan')):>6.3f}  "
                f"{row['run_dir']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize sampling sweep runs")
    parser.add_argument("--sweep-dir", type=str, default="models/sampling_sweeps", help="Sweep output directory")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        return 2

    rows = collect_rows(sweep_dir)
    write_outputs(rows, sweep_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
