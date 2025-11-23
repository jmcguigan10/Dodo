#!/usr/bin/env python3
"""
Parse training logs to plot train/val loss curves over epochs.

Assumes the training script logs lines of the form:
  train_loss=...,  val_loss=...
per epoch. If you want more granular curves, extend the training loop to emit
per-batch loss JSON lines and adjust this parser accordingly.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_PATTERN = re.compile(r"train_loss=([0-9eE\.\-+]+)\s+val_loss=([0-9eE\.\-+]+)")


def parse_log(log_path: Path) -> Tuple[List[float], List[float]]:
    train_losses = []
    val_losses = []
    for line in log_path.read_text().splitlines():
        m = EPOCH_PATTERN.search(line)
        if m:
            train_losses.append(float(m.group(1)))
            val_losses.append(float(m.group(2)))
    return train_losses, val_losses


def plot_curves(train_losses: List[float], val_losses: List[float], outdir: Path, title: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train_loss", marker="o")
    plt.plot(val_losses, label="val_loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpath = outdir / "train_val_curve.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from log file")
    parser.add_argument("--log", default="train.log", help="Path to training log file")
    parser.add_argument("--outdir", default="results/plots", help="Directory to save plot")
    args = parser.parse_args()

    log_path = Path(args.log)
    train_losses, val_losses = parse_log(log_path)
    if not train_losses:
        print("No epoch loss lines found in the log.")
        return
    plot_curves(train_losses, val_losses, Path(args.outdir), title=f"Loss curves from {log_path.name}")


if __name__ == "__main__":
    main()
