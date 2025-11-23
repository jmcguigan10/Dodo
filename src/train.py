import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import ExperimentConfig
from .data import build_dataloaders
from .losses import closure_loss
from .model import ResidualFFIModel
from .utils import get_device, load_checkpoint, save_checkpoint, set_seed


def _build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int, min_lr: float, max_lr: float) -> LambdaLR:
    base_lr = optimizer.defaults["lr"]
    max_lr = max_lr or base_lr

    def lr_lambda(step: int) -> float:
        if total_steps == 0:
            return 1.0
        if step < warmup_steps:
            lr = min_lr + (max_lr - min_lr) * (step + 1) / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (max_lr - min_lr) * cosine
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


def _move_batch(batch, device: torch.device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def _run_epoch(
    model: ResidualFFIModel,
    loader,
    device: torch.device,
    cfg: ExperimentConfig,
    optimizer: AdamW,
    scaler: GradScaler,
    scheduler: LambdaLR,
    start_step: int,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_count = 0
    step = start_step

    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"

    for batch_idx, batch in enumerate(loader):
        inv, resid_true, f_box_flat, f_true_flat = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            resid_pred_flat = model(inv)
            resid_pred = resid_pred_flat.view(-1, 6, 4)
            f_box = f_box_flat.view(-1, 6, 4)
            f_true = f_true_flat.view(-1, 6, 4)
            f_pred = f_box + resid_pred
            loss = closure_loss(f_pred, f_true, resid_pred_flat, resid_true, cfg.loss)

        scaler.scale(loss).backward()
        if cfg.training.grad_clip and cfg.training.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_size = inv.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_count += batch_size
        step += 1

        if cfg.training.log_interval and (batch_idx + 1) % cfg.training.log_interval == 0:
            avg = total_loss / max(total_count, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step}: loss={avg:.4f}, lr={current_lr:.3e}")

    return total_loss / max(total_count, 1), step


@torch.no_grad()
def _evaluate(model: ResidualFFIModel, loader, device: torch.device, cfg: ExperimentConfig) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        inv, resid_true, f_box_flat, f_true_flat = _move_batch(batch, device)
        resid_pred_flat = model(inv)
        resid_pred = resid_pred_flat.view(-1, 6, 4)
        f_box = f_box_flat.view(-1, 6, 4)
        f_true = f_true_flat.view(-1, 6, 4)
        f_pred = f_box + resid_pred
        loss = closure_loss(f_pred, f_true, resid_pred_flat, resid_true, cfg.loss)
        batch_size = inv.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def train_and_eval(cfg: ExperimentConfig) -> Dict[str, float]:
    set_seed(cfg.data.seed)
    cfg.ensure_directories()

    loaders, splits, _ = build_dataloaders(cfg.data)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset sizes → train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    model = ResidualFFIModel(**cfg.model.__dict__).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
    )

    total_steps = cfg.training.epochs * max(len(loaders["train"]), 1)
    scheduler = _build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=cfg.scheduler.warmup_steps,
        min_lr=cfg.scheduler.min_lr,
        max_lr=cfg.scheduler.max_lr or cfg.optimizer.lr,
    )

    scaler = GradScaler(enabled=cfg.training.mixed_precision and device.type == "cuda")

    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    best_ckpt_path = Path(cfg.training.checkpoint_dir) / "best.pt"
    final_ckpt_path = Path(cfg.training.checkpoint_dir) / "final.pt"

    # Resume if a checkpoint exists
    state = load_checkpoint(cfg.training.resume_from, map_location=device)
    if state is not None:
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state.get("epoch", 0) + 1
        best_val = state.get("best_val", best_val)
        global_step = state.get("global_step", 0)

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        train_loss, global_step = _run_epoch(
            model,
            loaders["train"],
            device,
            cfg,
            optimizer,
            scaler,
            scheduler,
            global_step,
        )
        val_loss = _evaluate(model, loaders["val"], device, cfg)
        print(f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val": best_val,
                },
                best_ckpt_path,
            )
            print(f"Saved improved checkpoint to {best_ckpt_path}")
        final_epoch = epoch

    test_loss = _evaluate(model, loaders["test"], device, cfg)
    # Save final checkpoint regardless of val performance
    save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": final_epoch,
            "global_step": global_step,
            "best_val": best_val,
        },
        final_ckpt_path,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results_path = Path(cfg.training.results_dir) / f"metrics_{timestamp}.json"
    metrics = {
        "val_loss": best_val,
        "test_loss": test_loss,
        "best_checkpoint": str(best_ckpt_path),
        "final_checkpoint": str(final_ckpt_path),
        "timestamp": timestamp,
        "config": {
            "data": cfg.data.__dict__,
            "model": cfg.model.__dict__,
            "optimizer": cfg.optimizer.__dict__,
            "scheduler": cfg.scheduler.__dict__,
            "training": {k: v for k, v in cfg.training.__dict__.items()},
            "loss": cfg.loss.__dict__,
        },
    }
    results_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Saved final checkpoint to {final_ckpt_path}")
    print(f"Saved metrics to {results_path}")
    return {"val_loss": best_val, "test_loss": test_loss}


def main():
    parser = argparse.ArgumentParser(description="Train residual-on-Box3D regression model")
    parser.add_argument("--config", default="config/model.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
