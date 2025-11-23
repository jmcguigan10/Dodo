#!/usr/bin/env python
# scripts/run_preprocess.py
#
# Thin Python wrapper around the Julia preprocessing script.
# It:
#   * loads the YAML config using OmegaConf (same config that training will use),
#   * ensures output directories exist,
#   * calls Julia with the right arguments.
#
import argparse
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dodo preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pre_process.yaml",
        help="Path to YAML config (shared with Julia and training scripts)",
    )
    parser.add_argument(
        "--julia",
        type=str,
        default="julia",
        help="Julia executable to use (default: 'julia' on PATH)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=".",
        help="Julia project path to use with --project (default: repo root)",
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    data_root = (repo_root / cfg.get("data_root", "data")).resolve()
    output_root = (repo_root / cfg.get("output_root", "pdata")).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[run_preprocess] config:      {config_path}")
    print(f"[run_preprocess] data_root:   {data_root}")
    print(f"[run_preprocess] output_root: {output_root}")

    julia_script = repo_root / "scripts" / "julia_preprocess.jl"
    if not julia_script.exists():
        raise SystemExit(f"Julia script not found at {julia_script}")

    cmd = [
        args.julia,
        f"--project={args.project}",
        str(julia_script),
        "--config",
        str(config_path),
    ]

    env = os.environ.copy()
    env.setdefault("JULIA_DEPOT_PATH", str(repo_root / ".julia_depot"))

    print(f"[run_preprocess] running: {' '.join(cmd)}")
    print(f"[run_preprocess] JULIA_DEPOT_PATH={env['JULIA_DEPOT_PATH']}")
    subprocess.run(cmd, check=True, env=env)
    print("[run_preprocess] preprocessing finished successfully")


if __name__ == "__main__":
    main()
