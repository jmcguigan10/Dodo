import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFFIModel(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, num_layers=3, output_dim=24):
        super().__init__()

        layers = []
        dim_in = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.LeakyReLU(0.01))
            dim_in = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch, 27) normalized invariants
        returns: (batch, 24) residuals ΔF_flat
        """
        h = self.trunk(x)
        ΔF_flat = self.out(h)
        return ΔF_flat
