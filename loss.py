def loss_fn(
    F_pred, F_true, c=1.0, eps=1e-8, w_n=1.0, w_mag=1.0, w_dir=1.0, w_unphys=100.0
):
    n_true = F_true[..., 0]
    n_pred = F_pred[..., 0]

    vec_true = F_true[..., 1:]
    vec_pred = F_pred[..., 1:]

    # Magnitudes
    mag_true = torch.linalg.norm(vec_true, dim=-1) + eps
    mag_pred = torch.linalg.norm(vec_pred, dim=-1)

    # Density loss
    L_n = ((n_pred - n_true) ** 2 / (n_true**2 + eps)).mean()

    # Flux magnitude loss
    L_mag = ((mag_pred - mag_true) ** 2 / (mag_true**2 + eps)).mean()

    # Direction loss
    u_true = vec_true / mag_true.unsqueeze(-1)
    u_pred = vec_pred / (mag_pred.unsqueeze(-1) + eps)
    cos = (u_true * u_pred).sum(dim=-1)
    L_dir = (1.0 - cos).mean()

    # Unphysical penalty
    fluxfac_pred = mag_pred / (c * n_pred + eps)
    L_unphys = torch.relu(fluxfac_pred - 1.0).mean()

    return w_n * L_n + w_mag * L_mag + w_dir * L_dir + w_unphys * L_unphys
