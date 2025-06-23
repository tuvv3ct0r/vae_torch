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
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return [recon_x, x, mu, log_var]
    
    def sample(self, num_samples) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(self.decoder.fc.weight.device)
        return self.decoder(z)
    
    def generate(self, z):
        return self.decoder(z)
    
    def loss_function(self, recon_x, x, mu, log_var, **kwargs):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.kld_weight * kl_loss
        if torch.isnan(loss):
            print(f"NaN loss detected: Recon={recon_loss.item()}, KLD={kl_loss.item()}, Batch={kwargs.get('batch_idx', 0)}")
        else:
            if kwargs.get('batch_idx', 0) == 0:
                print(f"Recon Loss: {recon_loss.item()}, KLD: {kl_loss.item()}, Total Loss: {loss.item()}")
        return {
            'loss': loss,
            'Reconstruction_Loss': recon_loss.detach(),
            'KLD': kl_loss.detach()
        }