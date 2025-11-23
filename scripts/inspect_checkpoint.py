#!/usr/bin/env python3
"""
Inspect a checkpoint by reporting loss term breakdowns on a small val/test slice.

Usage:
  python3 scripts/inspect_checkpoint.py --config config/model.yaml --split val --samples 256
"""

import argparse
from pathlib import Path

import torch

from src.config import ExperimentConfig
from src.data import build_dataloaders
from src.losses import closure_loss
from src.model import ResidualFFIModel


@torch.no_grad()
def inspect(cfg: ExperimentConfig, split: str, samples: int):
    # Avoid multiprocess issues for ad-hoc inspection
    cfg.data.num_workers = 0
    loaders, splits, norm_stats = build_dataloaders(cfg.data)
    if split not in loaders:
        raise ValueError(f"Split must be one of {list(loaders.keys())}")

    device = torch.device("cpu")
    ckpt = torch.load(Path(cfg.training.checkpoint_dir) / "best.pt", map_location=device)
    model = ResidualFFIModel(**cfg.model.__dict__)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total = 0
    agg = {"L_n": 0.0, "L_mag": 0.0, "L_dir": 0.0, "L_unphys": 0.0, "L_resid": 0.0, "total": 0.0}
    fluxfac_pred_vals = []
    fluxfac_true_vals = []

    for batch in loaders[split]:
        inv, resid_true, f_box_flat, f_true_flat = batch
        resid_pred_flat = model(inv)
        f_box = f_box_flat.view(-1, 6, 4)
        f_true = f_true_flat.view(-1, 6, 4)
        resid_pred = resid_pred_flat.view(-1, 6, 4)
        f_pred = f_box + resid_pred

        eps = cfg.loss.eps
        c = cfg.loss.speed_of_light
        n_true = f_true[..., 0]
        n_pred = f_pred[..., 0]
        vec_true = f_true[..., 1:]
        vec_pred = f_pred[..., 1:]
        mag_true = torch.linalg.norm(vec_true, dim=-1) + eps
        mag_pred = torch.linalg.norm(vec_pred, dim=-1)

        denom_n = torch.clamp(n_true.abs(), min=cfg.loss.density_floor) ** 2 + eps
        L_n = ((n_pred - n_true) ** 2 / denom_n).mean()
        L_mag = ((mag_pred - mag_true) ** 2 / (mag_true**2 + eps)).mean()
        u_true = vec_true / mag_true.unsqueeze(-1)
        u_pred = vec_pred / (mag_pred.unsqueeze(-1) + eps)
        cos = (u_true * u_pred).sum(dim=-1).clamp(-1.0, 1.0)
        L_dir = (1.0 - cos).mean()
        fluxfac_pred = mag_pred / (c * n_pred.abs() + eps)
        L_unphys = torch.relu(fluxfac_pred - 1.0).mean()
        L_resid = torch.nn.functional.l1_loss(resid_pred_flat, resid_true)
        total_loss = (
            cfg.loss.w_density * L_n
            + cfg.loss.w_magnitude * L_mag
            + cfg.loss.w_direction * L_dir
            + cfg.loss.w_unphysical * L_unphys
            + cfg.loss.w_residual_l1 * L_resid
        )

        bs = inv.shape[0]
        agg["L_n"] += L_n.item() * bs
        agg["L_mag"] += L_mag.item() * bs
        agg["L_dir"] += L_dir.item() * bs
        agg["L_unphys"] += L_unphys.item() * bs
        agg["L_resid"] += L_resid.item() * bs
        agg["total"] += total_loss.item() * bs
        fluxfac_pred_vals.append(fluxfac_pred.flatten())
        fluxfac_true_vals.append((mag_true / (n_true.abs() + eps)).flatten())
        total += bs

        if total >= samples:
            break

    for k in agg:
        agg[k] /= max(total, 1)
    fluxfac_pred_vals = torch.cat(fluxfac_pred_vals)[:samples]
    fluxfac_true_vals = torch.cat(fluxfac_true_vals)[:samples]

    print(f"Split: {split}, samples: {total}, target_scale: {norm_stats.get('target_scale'):.3e}")
    for k in ["total", "L_n", "L_mag", "L_dir", "L_unphys", "L_resid"]:
        print(f"{k}: {agg[k]:.4f}")
    print(
        f"Flux factor pred: mean {fluxfac_pred_vals.mean():.3f}, max {fluxfac_pred_vals.max():.3f} | "
        f"true: mean {fluxfac_true_vals.mean():.3f}, max {fluxfac_true_vals.max():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint loss breakdowns")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=256, help="Number of samples to aggregate (approx)")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    inspect(cfg, args.split, args.samples)


if __name__ == "__main__":
    main()
