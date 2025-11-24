#!/usr/bin/env python3
"""
Grid study runner that spawns multiple training runs with config overrides.

By default, it sweeps a modest grid over LR, hidden_dim, w_density, and w_residual_l1.
Each run gets isolated checkpoints/results under results/studies/<timestamp>/<run_id>/.

Usage:
  python3 scripts/run_study.py                       # run default grid
  python3 scripts/run_study.py --base-config config/model.yaml --tag my_study
"""

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


def build_run_config(base_cfg: Dict, overrides: Dict, run_dir: Path) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON

    # Apply overrides
    for key_path, value in overrides.items():
        cur = cfg
        for key in key_path[:-1]:
            cur = cur[key]
        cur[key_path[-1]] = value

    # Route outputs to run_dir
    cfg["training"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["training"]["results_dir"] = str(run_dir / "results")
    cfg["training"]["resume_from"] = None  # force fresh run

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run a grid of training jobs")
    parser.add_argument(
        "--base-config",
        default="config/model.yaml",
        help="Base YAML config to start from",
    )
    parser.add_argument("--tag", default=None, help="Optional study tag (folder name)")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base_config).read_text())

    # Define the grid here (edit as needed)
    lrs = [1e-3, 5e-4]
    hidden_dims = [192, 256, 320]
    w_density = [1.0, 0.5]
    w_resid = [0.2, 0.4]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    study_root = Path("results") / "studies" / (args.tag or f"study_{timestamp}")
    study_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict] = []
    run_id = 0
    for lr, hd, wd, wr in itertools.product(lrs, hidden_dims, w_density, w_resid):
        run_id += 1
        run_dir = study_root / f"run_{run_id:03d}_lr{lr}_hd{hd}_wd{wd}_wr{wr}"
        run_dir.mkdir(parents=True, exist_ok=True)
        overrides = {
            ("optimizer", "lr"): lr,
            ("model", "hidden_dim"): hd,
            ("loss", "w_density"): wd,
            ("loss", "w_residual_l1"): wr,
        }
        cfg = build_run_config(base_cfg, overrides, run_dir)
        cfg_path = run_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runs.append((run_dir, cfg_path))

    print(f"Prepared {len(runs)} runs under {study_root}")
    for i, (run_dir, cfg_path) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] starting {run_dir.name}")
        log_path = run_dir / "stdout.log"
        with log_path.open("w") as f:
            proc = subprocess.run(
                [sys.executable, "train.py", "--config", str(cfg_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parent.parent,
            )
        if proc.returncode == 0:
            print(f"{run_dir.name} completed")
        else:
            print(f"{run_dir.name} failed (see {log_path})")


if __name__ == "__main__":
    main()
