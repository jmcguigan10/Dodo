import torch
import torch.nn.functional as F

from .config import LossConfig


def closure_loss(
    F_pred: torch.Tensor,
    F_true: torch.Tensor,
    resid_pred: torch.Tensor,
    resid_true: torch.Tensor,
    cfg: LossConfig,
) -> torch.Tensor:
    """
    Composite loss that keeps densities/fluxes physical and anchors the residual.
    """
    eps = cfg.eps
    c = cfg.speed_of_light

    n_true = F_true[..., 0]
    n_pred = F_pred[..., 0]

    vec_true = F_true[..., 1:]
    vec_pred = F_pred[..., 1:]

    mag_true = torch.linalg.norm(vec_true, dim=-1) + eps
    mag_pred = torch.linalg.norm(vec_pred, dim=-1)

    L_n = ((n_pred - n_true) ** 2 / (n_true**2 + eps)).mean()

    L_mag = ((mag_pred - mag_true) ** 2 / (mag_true**2 + eps)).mean()

    u_true = vec_true / mag_true.unsqueeze(-1)
    u_pred = vec_pred / (mag_pred.unsqueeze(-1) + eps)
    cos = (u_true * u_pred).sum(dim=-1).clamp(-1.0, 1.0)
    L_dir = (1.0 - cos).mean()

    fluxfac_pred = mag_pred / (c * n_pred + eps)
    L_unphys = torch.relu(fluxfac_pred - 1.0).mean()

    L_resid = F.l1_loss(resid_pred, resid_true)

    return (
        cfg.w_density * L_n
        + cfg.w_magnitude * L_mag
        + cfg.w_direction * L_dir
        + cfg.w_unphysical * L_unphys
        + cfg.w_residual_l1 * L_resid
    )
