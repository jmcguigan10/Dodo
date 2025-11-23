import torch
from torch.utils.data import Dataset


class EmuResidualDataset(Dataset):
    def __init__(self, h5_path, indices, norm_stats):
        self.f = h5py.File(h5_path, "r")
        self.indices = indices
        self.mean_inv = norm_stats["mean_inv"]
        self.std_inv = norm_stats["std_inv"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        k = self.indices[idx]
        inv = self.f["invariants"][k]  # (27,)
        resid = self.f["residual"][k]  # (24,)
        F_box = self.f["F_box"][k].reshape(-1)  # (24,)
        F_true = self.f["F_true"][k].reshape(-1)

        inv_norm = (inv - self.mean_inv) / (self.std_inv + 1e-8)

        return (
            torch.from_numpy(inv_norm).float(),
            torch.from_numpy(resid).float(),
            torch.from_numpy(F_box).float(),
            torch.from_numpy(F_true).float(),
        )
