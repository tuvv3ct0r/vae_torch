import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], latent_dim=128):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, hidden_dims[3] * 4 * 4)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[3], hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[0], in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.fc(z)
        result = result.view(result.shape[0], self.hidden_dims[3], 4, 4)
        result = self.conv_block(result)
        return result