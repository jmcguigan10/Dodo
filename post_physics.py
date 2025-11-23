import torch


def project_to_physical(F_pred, F_init, c=1.0, eps=1e-8):
    """
    Enforce basic physical constraints:
    - non-negative densities
    - flux factor <= 1
    Optionally: adjust densities to conserve ELN.

    F_pred: (B, 6, 4)
    F_init: (B, 6, 4)  # can be used for ELN constraints
    """
    F = F_pred.clone()
    # assume index 0 is time component; adjust if needed
    n = F[..., 0]
    vec = F[..., 1:]  # (B, 6, 3)

    # Non-negative densities
    n = torch.clamp(n, min=eps)
    F[..., 0] = n

    # Flux factor <= 1
    mag = torch.linalg.norm(vec, dim=-1)  # (B, 6)
    limit = c * n + eps
    mask = mag > limit
    scale = limit / (mag + eps)
    scale = torch.where(mask, scale, torch.ones_like(scale))
    F[..., 1:] = vec * scale.unsqueeze(-1)

    # (Optional) enforce ELN conservation via small correction to n

    return F
