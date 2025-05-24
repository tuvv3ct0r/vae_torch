import torch
from torch import nn
from torch.nn import functional as F

from .decoder import Decoder
from .encoder import Encoder


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256], kld_weight=0.00025, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight

        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims, latent_dim)

    def reparameterize(self, mu, log_var) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), x, mu, log_var]
    
    def sample(self, num_samples) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(self.decoder.fc.weight.device)
        return self.decoder(z)
    
    def generate(self, z):
        return self.decoder(z)
    
    def loss_function(self, recon_x, x, mu, log_var, **kwargs):
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.kld_weight * kl_loss
        return {'loss': loss, 'Reconstruction_Loss': recon_loss.detach(), 'KLD': kl_loss.detach()}