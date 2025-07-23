import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional


class VAEEncoder(nn.Module):
    def __init__(self, model: nn.Module, z_dim: int):
        super().__init__()
        self.model = model
        self.z_proj = nn.Linear(model.output_dim, 2 * z_dim)

    def forward(self, x):
        x = self.model(x)
        mean, logvar = torch.chunk(self.z_proj(x), 2, dim=-1)
        return mean, logvar


class VAEDecoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, z, batch):
        x_hat = self.model(z, batch)
        return x_hat


class BetaVaeModel(nn.Module):
    def __init__(self, encoder: VAEEncoder, decoder: VAEDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        mean, logvar = self.encoder(batch)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_hat = self.decoder(z, batch)
        return x_hat, mean, logvar

    def encode(self, batch):
        mean, _ = self.encoder(batch)
        return mean

    def decode(self, z, batch):
        return self.decoder(z, batch)


class BetaVAE:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        z_dim: int,
        beta: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        lr: float = 1e-3,
    ):
        self.model = BetaVaeModel(VAEEncoder(encoder, z_dim), VAEDecoder(decoder))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.beta = beta
        self.weights = weights or {}

    def loss_fn(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_hat, mean, logvar = self.model(batch)
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1).mean()

        recon_losses = {}
        total_recon = 0.0

        for key in batch:
            if key == 'mask':
                continue
            x = batch[key]
            x_hat_k = x_hat[key]
            if key == 'action':
                loss = ((x - x_hat_k) ** 2).sum(dim=-1)
                loss = (loss * batch['mask']).sum(dim=-1).mean()
            else:
                loss = ((x - x_hat_k) ** 2).view(x.size(0), -1).sum(dim=-1).mean()
            weighted_loss = self.weights.get(key, 1.0) * loss
            recon_losses[f'recon_loss/{key}'] = weighted_loss
            total_recon += weighted_loss

        total_loss = total_recon + self.beta * kl
        recon_losses['kl_loss'] = kl
        recon_losses['recon_loss/total'] = total_recon
        recon_losses['loss'] = total_loss
        return total_loss, recon_losses

    def train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self.model.train()
        self.optimizer.zero_grad()
        loss, info = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        return info

    def val_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            _, info = self.loss_fn(batch)
        return info

    def predict(self, batch: Dict) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(batch)
