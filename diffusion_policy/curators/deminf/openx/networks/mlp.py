from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        activate_final: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i, dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(
                in_features=hidden_dims[i - 1] if i > 0 else None,  # will set later in forward
                out_features=dim
            ))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(dim))
            else:
                self.norms.append(None)
            if dropout_rate is not None and dropout_rate > 0 and i + 1 < len(hidden_dims):
                self.dropouts.append(nn.Dropout(p=dropout_rate))
            else:
                self.dropouts.append(None)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            # Handle first layer input size setup
            if layer.in_features is None:
                self.layers[i] = nn.Linear(x.shape[-1], layer.out_features).to(x.device)
                layer = self.layers[i]

            x = layer(x)
            is_final = (i == len(self.layers) - 1)
            if not is_final or self.activate_final:
                if self.use_layer_norm:
                    x = self.norms[i](x)
                if self.dropouts[i] is not None:
                    x = self.dropouts[i](x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(
        self,
        features: int,
        act: Callable[[torch.Tensor], torch.Tensor],
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.act = act
        self.use_layer_norm = use_layer_norm
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else None
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.ln = nn.LayerNorm(features) if use_layer_norm else None
        self.res_proj = nn.Linear(features, features)

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        residual = x
        if self.use_layer_norm:
            x = self.ln(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        if residual.shape != x.shape:
            residual = self.res_proj(residual)
        return residual + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, features: int, learned: bool = False, scale: float = 10000.0):
        super().__init__()
        self.features = features
        self.learned = learned
        self.scale = scale
        if learned:
            self.weight = nn.Parameter(torch.randn(features // 2, 1) * 0.2)
        else:
            self.register_buffer("inv_freq", torch.exp(
                -math.log(scale) * torch.arange(0, features // 2, dtype=torch.float32) / (features // 2 - 1)
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned:
            emb = 2 * math.pi * torch.matmul(x, self.weight.T)
        else:
            # Assume x: (B, 1) or (B,)
            x = x.unsqueeze(-1) if x.ndim == 1 else x
            emb = x * self.inv_freq
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activation: Callable = F.silu,
        time_features: int = 64,
        learn_time_embedding: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_features, learned=learn_time_embedding),
            MLP(
                hidden_dims=[2 * time_features, time_features],
                activation=activation,
                activate_final=False
            )
        )

        self.input_proj = nn.Linear(None, hidden_dim)  # Set in forward

        self.blocks = nn.ModuleList([
            MLPResNetBlock(
                hidden_dim,
                act=activation,
                use_layer_norm=use_layer_norm,
                dropout_rate=dropout_rate
            )
            for _ in range(num_blocks)
        ])

    def forward(self, obs: torch.Tensor, action: torch.Tensor, time: torch.Tensor, train: bool = False) -> torch.Tensor:
        # time: (B,) or (B, 1)
        t = self.time_emb(time)

        # action: (B, H, D)
        b, h, d = action.shape
        action = action.view(b, h * d)

        x = torch.cat([obs, action, t], dim=-1)

        if isinstance(self.input_proj, nn.Linear) and self.input_proj.in_features is None:
            self.input_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, train=train)

        return self.activation(x)
