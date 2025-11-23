# ML Closure for Fast Neutrino Flavor Instability  
## Residual-on-Box3D Pipeline & Architecture

This document describes an end‑to‑end pipeline to train a machine‑learning closure for fast neutrino flavor instability (FFI) using the **Zenodo Rhea datasets** and an **analytic Box3D baseline**.

The goal is to learn a **local, instantaneous mapping** from initial neutrino radiation moments to their **asymptotic post‑FFI moments**, using ML **only to predict residual corrections** on top of a physically motivated analytic model (Box3D).

---

## 1. Problem Statement

We consider, at each spacetime point in a neutron‑star merger simulation, the neutrino number four‑fluxes
$$
F_{a}^{\mu} \quad\text{for}\quad a = 1,\dots,6
$$
where:
- $a$ indexes the six species  
  $\{\nu_e,\ \bar\nu_e,\ \nu_\mu,\ \bar\nu_\mu,\ \nu_\tau,\ \bar\nu_\tau\}$,
- \(\mu\) is the spacetime index (4 components in an orthonormal tetrad).

Given:
- initial four‑fluxes $F_{a,\text{init}}^{\mu}$,
- the local fluid four‑velocity $u^\mu$,

we want a closure that predicts the **asymptotic four‑fluxes** after fast flavor conversion:
$$
F_{a,\text{final}}^{\mu} \approx \hat{F}_{a}^{\mu}(F_{\text{init}}, u^\mu).
$$

Instead of having the neural network approximate $\hat{F}$ directly, we:
1. Use an analytic scheme (Box3D) to compute a **baseline prediction** $F_{a,\text{Box3D}}^{\mu}$.
2. Train the network to predict the **residual**:
   $$
   \Delta F_a^\mu = F_{a,\text{Emu}}^\mu - F_{a,\text{Box3D}}^\mu,
   $$
   where $F_{a,\text{Emu}}^\mu$ is the asymptotic state from QKE (Emu) simulations.

At inference:
$$
F_{a,\text{pred}}^\mu = F_{a,\text{Box3D}}^\mu + \Delta F_{a,\text{ML}}^\mu.
$$

This makes the ML correction a small, physics‑constrained adjustment rather than the entire closure.

---

## 2. Data Sources (Zenodo)

We assume access to the Zenodo Rhea datasets (as described in the original paper):

### 2.1 Emu_data (local QKE simulations)

For each local QKE run `k`:

- `F4_initial[k, 4, 2, 3]`  
- `F4_final[k, 4, 2, 3]`  
- `growthRate[k]`  
- possibly standard deviations and diagnostics.

Where the axes correspond to:
- 4 spacetime components,
- 2 species (ν vs $\bar\nu$),
- 3 flavors (e, μ, τ).

We convert these to a consistent shape:
$$
F_{\text{init}}[k] \in \mathbb{R}^{6\times 4}, \qquad
F_{\text{true}}[k] \in \mathbb{R}^{6\times 4}.
$$

### 2.2 SpEC_data (M1 snapshots) — optional for v1

SpEC snapshots contain:
- M1‑level moments (`n_e`, `n_a`, `n_x`, flux vectors),
- matter fields $(\rho, T, Y_e)$,
- ELN crossing diagnostics.

For the **first ML model**, you can train entirely on Emu_data, and use SpEC_data later for testing/integration.

---

## 3. Baseline: Analytic Box3D Closure

We define a function
$$
\mathcal{B} : F_{\text{init}} \mapsto F_{\text{Box3D}}
$$
which implements the **Box3D analytic FFI closure** described in the paper.

### 3.1 Conceptual steps inside Box3D

For each sample:

1. Reconstruct flavor‑dependent angular distributions $f_\alpha(\hat{n})$ from the initial moments via a maximum‑entropy closure that matches:
   - number densities,
   - flux vectors.

2. Construct ELN and XLN angular distributions from these $f_\alpha(\hat{n})$.

3. Solve integral constraints that enforce:
   - conservation of relevant charges (e.g. total lepton number),
   - elimination of ELN‑XLN crossings in the asymptotic state.

4. Define a piecewise survival probability $P(\hat{n})$ for each angular region.

5. Compute the asymptotic flavor distributions $f'_\alpha(\hat{n})$ and integrate over solid angle to obtain the **final four‑fluxes**:
   $$
   F_{a,\text{Box3D}}^\mu = \int f'_{\alpha(a)}(\hat{n}) \, p^\mu(\hat{n}) \, d\Omega.
   $$

In code, this is a pure deterministic function executed **offline** before training.

### 3.2 Implementation sketch (offline)

```python
def box3d_closure(F_init, matter_fields, params):
    """
    F_init: np.ndarray, shape (6, 4)  # 6 species x 4 components
    matter_fields: e.g. rho, T, Ye, etc. if required
    params: config for angular resolution, tolerance, etc.

    Returns:
        F_box: np.ndarray, shape (6, 4)
    """
    # 1. Reconstruct angular distributions f_alpha(nhat)
    # 2. Compute ELN/XLN angular profiles
    # 3. Solve Box3D constraints for survival probabilities
    # 4. Integrate to get F_box

    # Pseudocode only – fill in with the actual Box3D formulas
    F_box = np.zeros_like(F_init)
    # ...
    return F_box
```
```python
import numpy as np

def compute_invariants(F_init, u4, metric=None):
    """
    F_init: np.ndarray, shape (6, 4)
    u4:     np.ndarray, shape (4,)
    metric: np.ndarray, shape (4, 4), Minkowski metric

    Returns:
        invariants: np.ndarray, shape (27,)
    """
    if metric is None:
        # Example Minkowski metric with signature (-,+,+,+)
        metric = np.diag([-1.0, 1.0, 1.0, 1.0])

    inv = []
    # 21 F·F invariants
    for a in range(6):
        Fa = F_init[a]                 # (4,)
        Fa_g = Fa @ metric             # (4,)
        for b in range(a, 6):
            Fb = F_init[b]             # (4,)
            inv.append(Fa_g @ Fb)      # scalar

    # 6 F·u invariants
    u_g = u4 @ metric
    for a in range(6):
        Fa = F_init[a]
        inv.append(Fa @ u_g)

    return np.array(inv, dtype=np.float32)  # (27,)
```