import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], latent_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[1], hidden_dims[3], kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(hidden_dims[3]),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dims[3] * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[3] * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv_block(x)
        x = x.view(x.shape[0], -1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var