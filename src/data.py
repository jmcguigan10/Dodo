from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .utils import set_seed


def _feature_sample_axis(shape, feature_dim: int) -> int:
    if len(shape) != 2:
        raise ValueError(f"Expected 2-D feature dataset, got shape {shape}")
    if shape[0] == feature_dim and shape[1] != feature_dim:
        return 1
    if shape[1] == feature_dim and shape[0] != feature_dim:
        return 0
    return 1 if shape[1] >= shape[0] else 0


def _flux_axes(shape) -> Tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected 3-D flux dataset, got shape {shape}")
    axes = list(range(len(shape)))
    species_axis = shape.index(6) if 6 in shape else None
    comp_axis = shape.index(4) if 4 in shape else None
    if species_axis is None or comp_axis is None:
        raise ValueError(f"Flux dataset must contain dimensions 6 and 4, got shape {shape}")
    sample_axes = [ax for ax in axes if ax not in (species_axis, comp_axis)]
    if len(sample_axes) != 1:
        raise ValueError(f"Could not infer sample axis for shape {shape}")
    return sample_axes[0], species_axis, comp_axis


def _slice_features(ds, idx: int, sample_axis: int) -> np.ndarray:
    if sample_axis == 0:
        arr = ds[idx, :]
    else:
        arr = ds[:, idx]
    return np.asarray(arr, dtype=np.float32)


def _slice_flux(ds, idx: int, sample_axis: int, species_axis: int, comp_axis: int) -> np.ndarray:
    slc = [slice(None)] * ds.ndim
    slc[sample_axis] = idx
    arr = np.asarray(ds[tuple(slc)], dtype=np.float32)

    remaining_axes = [ax for ax in range(ds.ndim) if ax != sample_axis]
    species_pos = remaining_axes.index(species_axis)
    comp_pos = remaining_axes.index(comp_axis)
    arr = np.transpose(arr, (species_pos, comp_pos))
    return arr  # (6, 4)


class EmuResidualDataset(Dataset):
    """
    Lazy dataset that serves normalized invariants and Box3D residual targets.
    """

    def __init__(self, h5_path: Path, indices: Iterable[int], norm_stats: Dict[str, np.ndarray]):
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.mean_inv = np.asarray(norm_stats["mean_inv"], dtype=np.float32)
        self.std_inv = np.asarray(norm_stats["std_inv"], dtype=np.float32)
        self._file = None
        with h5py.File(self.h5_path, "r") as f:
            self.inv_sample_axis = _feature_sample_axis(f["invariants"].shape, feature_dim=27)
            flux_shape = f["F_box"].shape
            self.sample_axis_flux, self.species_axis, self.comp_axis = _flux_axes(flux_shape)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int):
        f = self._get_file()
        k = int(self.indices[idx])

        inv = _slice_features(f["invariants"], k, self.inv_sample_axis)  # (27,)
        f_box = _slice_flux(f["F_box"], k, self.sample_axis_flux, self.species_axis, self.comp_axis)  # (6,4)
        f_true = _slice_flux(f["F_true"], k, self.sample_axis_flux, self.species_axis, self.comp_axis)  # (6,4)
        resid = (f_true - f_box).reshape(-1)  # consistent residual target

        inv_norm = (inv - self.mean_inv) / (self.std_inv + 1e-8)

        return (
            torch.from_numpy(inv_norm.astype(np.float32)),
            torch.from_numpy(resid.astype(np.float32)),
            torch.from_numpy(f_box.reshape(-1).astype(np.float32)),
            torch.from_numpy(f_true.reshape(-1).astype(np.float32)),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        if self._file is not None:
            self._file.close()


def _count_samples(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        inv_shape = f["invariants"].shape
        sample_axis = _feature_sample_axis(inv_shape, feature_dim=27)
        return inv_shape[sample_axis]


def _split_indices(n: int, val_split: float, test_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed(seed)
    all_idx = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)

    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_idx = all_idx[:n_test]
    val_idx = all_idx[n_test : n_test + n_val]
    train_idx = all_idx[n_test + n_val :]

    return train_idx, val_idx, test_idx


def _compute_norm_stats(h5_path: Path, indices: np.ndarray, chunk: int = 2048) -> Dict[str, np.ndarray]:
    sum_x = None
    sum_x2 = None
    total = 0

    with h5py.File(h5_path, "r") as f:
        ds = f["invariants"]
        dim = 27
        sample_axis = _feature_sample_axis(ds.shape, feature_dim=dim)
        sum_x = np.zeros(dim, dtype=np.float64)
        sum_x2 = np.zeros(dim, dtype=np.float64)

        for start in range(0, len(indices), chunk):
            idx_chunk = indices[start : start + chunk]
            idx_sorted = np.sort(idx_chunk)
            if sample_axis == 0:
                batch = np.asarray(ds[idx_sorted, :], dtype=np.float64)
            else:
                batch = np.asarray(ds[:, idx_sorted], dtype=np.float64).T
            sum_x += batch.sum(axis=0)
            sum_x2 += (batch**2).sum(axis=0)
            total += batch.shape[0]

    mean = sum_x / max(total, 1)
    var = sum_x2 / max(total, 1) - mean**2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return {"mean_inv": mean.astype(np.float32), "std_inv": std.astype(np.float32)}


def build_dataloaders(cfg: DataConfig) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    h5_path = Path(cfg.processed_path)
    n_samples = _count_samples(h5_path)
    train_idx, val_idx, test_idx = _split_indices(n_samples, cfg.val_split, cfg.test_split, cfg.seed)
    norm_stats = _compute_norm_stats(h5_path, train_idx)

    datasets = {
        "train": EmuResidualDataset(h5_path, train_idx, norm_stats),
        "val": EmuResidualDataset(h5_path, val_idx, norm_stats),
        "test": EmuResidualDataset(h5_path, test_idx, norm_stats),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last and split == "train",
        )
        for split, ds in datasets.items()
    }

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return loaders, splits, norm_stats
