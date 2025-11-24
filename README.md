# Dodo: Residual-on-Box3D ML Closure for Fast Neutrino Flavor Instability

This repo trains a machine-learning closure for fast neutrino flavor instability (FFI) using **Emu** local QKE simulations and a **Box3D** analytic baseline. The model learns only the **residual correction** on top of Box3D rather than the full closure.

High‑level mapping at each spacetime point:

- Inputs: initial four‑fluxes $F_{\text{init}} \in \mathbb{R}^{6\times 4}$ plus fluid four‑velocity $u^\mu$.
- Baseline: Box3D closure gives $F_{\text{Box3D}}$.
- Target: Emu asymptotic fluxes $F_{\text{true}}$.
- Network prediction: residual $\Delta F_{\text{ML}}$ such that
  $F_{\text{pred}} = F_{\text{Box3D}} + \Delta F_{\text{ML}}(F_{\text{inv}}),$
  where $F_{\text{inv}}$ are invariant features built from $F_{\text{init}}$ and $u^\mu$.

See `plan.md` for the physics/math rationale and `Notes.md` for the Box3D details.

---

## 1. Data and Preprocessing

Raw HDF5 data live in `data/`, with configuration in `config/pre_process.yaml`. Preprocessing is handled by Julia/Python glue:

- `scripts/Box3D.jl`: implementation of the Box3D analytic mixing scheme (Richers et al. 2025).
- `scripts/julia_preprocess.jl` and `scripts/get_cfg.jl`: Julia helpers for running Box3D and reading config.
- `scripts/run_preprocess.py`: Python driver that:
  - Reads raw Emu/SpEC HDF5 files according to `config/pre_process.yaml`.
  - Calls the Julia Box3D closure on each sample to get $F_{\text{Box3D}}$.
  - Computes invariants (see `plan.md` and `mk_invariant.jl`).
  - Writes a consolidated HDF5 file to `pdata/preprocessed_all.h5`.
  - Also computes target scaling statistics (99th percentile) used downstream.

The preprocessed file has (axes inferred automatically in the loader):

- `F_init`: initial fluxes, effectively shape `(N, 6, 4)` after axis reordering.
- `F_true`: Emu asymptotic fluxes `(N, 6, 4)`.
- `F_box`: Box3D baseline fluxes `(N, 6, 4)`.
- `invariants`: invariant features `(N, 27)`.
- `residual`: stored residuals `(N, 24)` (not strictly required for training but kept for inspection).
- `dirname`, `sim_id`: metadata about the originating simulation.

### Running preprocessing

- With `just` (preferred):

```sh
just preprocess                # uses config/pre_process.yaml
just preprocess config=...     # override config
```

- Directly:

```sh
python3 scripts/run_preprocess.py --config config/pre_process.yaml --julia julia
```
- Inspection:
```sh
python3 -m scripts.inspect_checkpoint --config config/model.yaml --split val --samples 512
```
- Metrics plot: 
```sh
python3 -m scripts.plot_metrics --results-dir results --outdir results/plots
```
- Train/val plot (after next run): 
```sh
python3 -m scripts.plot_training_curve --log results/train.log --outdir results/plots
```

This populates `pdata/preprocessed_all.h5`, which the training pipeline consumes.

---

## 2. Training Data Pipeline (`src/data.py`)

The training loader builds a residual dataset from `pdata/preprocessed_all.h5`:

- `EmuResidualDataset`:
  - opens the HDF5 file per worker.
  - Automatically infers sample/species/component axes for `F_box`, `F_true`, and `invariants`.
  - For each index $k$:
    - Reads invariants $x_k \in \mathbb{R}^{27}$.
    - Reorders `F_box[k]`, `F_true[k]` to `(6, 4)`.
    - Computes residual \(\Delta F_k = F_{\text{true},k} - F_{\text{Box3D},k}\) and flattens to 24 components.

- **Normalization and scaling**:
  - Compute `mean_inv`, `std_inv` over the **training split only** and normalize invariants as
    $(x - \mu)/\sigma$.
  - Compute a robust `target_scale`:
    - Default: 99th percentile of $|F_{\text{true}}|$ (across all components); for this dataset, $\sim 1.46\times10^{33}$.
    - Optional override: `data.target_scale` in `config/model.yaml`.
  - Divide `F_box`, `F_true`, and residuals by `target_scale` before returning them to keep values in a numerically stable range.

- Splitting:
  - Global index set is randomly shuffled with a fixed seed.
  - Split into train/val/test according to `data.val_split` and `data.test_split` in `config/model.yaml`.

The `build_dataloaders` function returns:

- A dict of `DataLoader`s for `"train"`, `"val"`, `"test"`.
- The index splits.
- Normalization stats including `target_scale`.

---

## 3. Model Architecture (`src/model.py`)

The core model is `ResidualFFIModel`, a fully connected residual network:

- Inputs: 27 normalized invariants per sample.
- Outputs: 24 residual components $\Delta F_{\text{ML}}$ (flattened `(6,4)`).
- Structure:
  - Input linear layer to a hidden dimension.
  - `hidden_layers` repeated residual blocks:
    - `Linear(dim_in, hidden_dim)`
    - Optional `BatchNorm1d`
    - Activation (`gelu` by default; configurable)
    - Optional dropout
    - Skip connection when `dim_in == hidden_dim`.
  - Final linear head to `output_dim = 24`.
- Initialization:
  - Kai-ming initialization for linear layers.
  - Batch norm weights initialized to 1, biases to 0.

This design is lightweight, stable, and expressive enough to learn small corrections on top of a strong physics prior (Box3D).

---

## 4. Loss Function (`src/losses.py`)

`closure_loss` enforces both numerical accuracy and basic physical constraints:

- Inputs per batch:
  - `F_pred`: predicted final fluxes (Box3D + ML residual), scaled by `target_scale`.
  - `F_true`: true asymptotic fluxes (scaled).
  - `resid_pred`: predicted residual (24‑vector).
  - `resid_true`: target residual (24‑vector).

- Terms:
  - **Density loss** (`L_n`): relative squared error on the time component with a floor on $|n_{\text{true}}|$ to prevent blowups when density is tiny (see `loss.density_floor`).
  - **Flux magnitude loss** (`L_mag`): relative squared error on $|\vec{F}|$.
  - **Direction loss** (`L_dir`): penalizes misalignment between flux direction unit vectors.
  - **Unphysical penalty** (`L_unphys`): penalizes flux factors $|\vec{F}|/(c |n|)$ that exceed 1.
  - **Residual L1 loss** (`L_resid`): $L^1$ between `resid_pred` and `resid_true`.

- Combined with weights from the `loss` block in `config/model.yaml`:

```yaml
loss:
  w_density: 1.0
  w_magnitude: 1.0
  w_direction: 1.0
  w_unphysical: 25.0
  w_residual_l1: 0.2
  density_floor: 0.1
  speed_of_light: 1.0
  eps: 1.0e-8
```

---

## 5. Training Loop (`src/train.py`)

`train_and_eval` orchestrates training, validation, and testing:

- Device and AMP:
  - Uses CUDA if available (`get_device()`), otherwise CPU.
  - Optional mixed precision with `torch.cuda.amp` when on GPU.

- Optimizer and scheduler:
  - `AdamW` with hyperparameters from `optimizer` in `config/model.yaml`.
  - Warmup + cosine learning rate schedule (`scheduler` block).

- Epoch loop:
  - For each epoch:
    - Train over the `"train"` loader; log running loss and LR every `training.log_interval` steps.
    - Evaluate on `"val"` loader.
    - Save `checkpoints/best.pt` whenever `val_loss` improves.
  - After all epochs:
    - Evaluate on `"test"` loader.
    - Save `checkpoints/final.pt` with final model, optimizer, scheduler, and scaler state.

- Outputs:
  - Checkpoints in `checkpoints/`:
    - `best.pt`: best validation loss.
    - `final.pt`: final epoch state.
  - Metrics in `results/metrics_<timestamp>.json`:
    - `val_loss`, `test_loss`.
    - Paths to best/final checkpoints.
    - Full config snapshot and `target_scale`.

You can resume from a checkpoint by setting `training.resume_from` in `config/model.yaml` to `checkpoints/best.pt` or `checkpoints/final.pt`.

---

## 6. Configuration (`src/config.py`, `config/model.yaml`)

`ExperimentConfig` bundles all configuration:

- `data`:

```yaml
data:
  processed_path: "pdata/preprocessed_all.h5"
  upload_dir: "data/uploads"
  batch_size: 256
  num_workers: 4
  val_split: 0.1
  test_split: 0.1
  seed: 1337
  pin_memory: true
  drop_last: false
  target_scale: null  # auto; or set, e.g. 1.0e33
```

- `model`: network width/depth, activation, dropout.
- `optimizer`: AdamW hyperparameters.
- `scheduler`: warmup and LR range.
- `training`: epochs, grad clipping, AMP, logging, checkpoint/results directories.
- `loss`: weights and numerical constants.

`ExperimentConfig.load(path)` reads YAML, fills defaults, and ensures required directories exist.

---

## 7. Running Training

### Environment

Recommended steps:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install torch h5py pyyaml numpy
```

### Commands

- Preprocess (once per dataset/config):

```sh
just preprocess
```

- Train the residual model:

```sh
just train                           # uses config/model.yaml
just train config=path/to/other.yaml
```

call the entrypoints directly:

```sh
python3 scripts/run_preprocess.py --config config/pre_process.yaml --julia julia
python3 train.py --config config/model.yaml
```

---

## 8. Why Residual-on-Box3D?

The key design choice is to let the neural network **correct** a robust analytic closure rather than replace it:

- **Physics prior**: Box3D encodes charge conservation, ELN/XLN crossing removal, and asymptotic flavor mixing structure. This keeps the baseline prediction close to physically allowed states.
- **Smaller dynamic range**: $\Delta F = F_{\text{true}} - F_{\text{Box3D}}$ is typically much smaller than either term individually, especially after scaling by `target_scale`. This makes the regression numerically easier.
- **Better extrapolation**: When the ML model encounters out‑of‑distribution inputs, Box3D still provides a physically sensible baseline; the residual network is encouraged not to make large, unconstrained jumps.
- **Invariant features**: Using invariants built from $F_{\text{init}}$ and $u^\mu$ focuses the model on physically meaningful degrees of freedom and reduces dependence on coordinate choices.

Overall, the pipeline is designed so that:

- Preprocessing uses Box3D and invariants to compress the physics into a structured training set.
- The model learns a small, constrained residual in a well‑scaled space.
- The loss function enforces both accuracy and basic physicality of the final fluxes.

This provides a robust starting point for exploring alternative architectures, loss terms, or additional inputs (e.g. matter fields) without changing the basic residual‑on‑Box3D philosophy.

---

## 9. Utilities and Study Runner

- `scripts/inspect_checkpoint.py`: print loss term breakdowns and flux-factor stats on a slice of train/val/test.
- `scripts/plot_metrics.py`: plot val/test loss across runs (`results/metrics_*.json`) to `results/plots/`.
- `scripts/plot_training_curve.py`: plot train/val curves from `results/train.log` (written by the training loop).
- `scripts/run_study.py`: grid-study runner that spawns multiple training runs with overrides (LR, hidden_dim, w_density, w_residual_l1) and isolates outputs under `results/studies/<tag>/run_XXX/`.
