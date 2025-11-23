#!/usr/bin/env python3
"""
Plot training/validation/test metrics across runs.

Usage:
  python3 scripts/plot_metrics.py --results-dir results --outdir results/plots

This reads all metrics_*.json files in the results directory and plots val/test loss
versus timestamp. If only one file is present, it still saves the plot.
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(results_dir: Path) -> List[dict]:
    metrics = []
    for path in sorted(results_dir.glob("metrics_*.json")):
        try:
            data = json.loads(path.read_text())
            data["_path"] = path
            metrics.append(data)
        except Exception:
            continue
    return metrics


def plot_losses(metrics: List[dict], outdir: Path) -> None:
    if not metrics:
        print("No metrics_*.json files found; nothing to plot.")
        return

    timestamps = [m.get("timestamp", m["_path"].stem.replace("metrics_", "")) for m in metrics]
    val = [m.get("val_loss", None) for m in metrics]
    test = [m.get("test_loss", None) for m in metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, val, marker="o", label="val_loss")
    plt.plot(timestamps, test, marker="s", label="test_loss")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Loss")
    plt.title("Validation/Test Loss across runs")
    plt.tight_layout()
    plt.legend()
    outpath = outdir / "loss_over_runs.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics_*.json files")
    parser.add_argument("--results-dir", default="results", help="Directory containing metrics_*.json")
    parser.add_argument("--outdir", default="results/plots", help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    metrics = load_metrics(results_dir)
    plot_losses(metrics, outdir)


if __name__ == "__main__":
    main()
