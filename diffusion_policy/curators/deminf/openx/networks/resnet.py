import math
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU, transpose=False):
        super().__init__()
        self.act = activation()
        self.transpose = transpose

        if transpose:
            self.up = nn.Upsample(scale_factor=stride, mode='nearest') if stride > 1 else nn.Identity()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)

        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)

        self.need_proj = stride != 1 or in_channels != out_channels
        if self.need_proj:
            self.proj = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest') if transpose and stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        residual = x
        if self.transpose:
            x = self.up(x)

        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        if self.need_proj:
            residual = self.proj(residual)

        return self.act(x + residual)


class BottleneckResNetBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, stride=1, activation=nn.ReLU):
        super().__init__()
        self.act = activation()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(32, bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels * 4, 1, bias=False)
        self.norm3 = nn.GroupNorm(32, bottleneck_channels * 4)

        out_channels = bottleneck_channels * 4
        self.need_proj = stride != 1 or in_channels != out_channels
        if self.need_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        if self.need_proj:
            residual = self.proj(residual)
        return self.act(x + residual)


class SpatialSoftmax(nn.Module):
    def __init__(self, num_kp=64, temperature=1.0):
        super().__init__()
        self.num_kp = num_kp
        self.temperature = temperature
        self.conv = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv is None:
            self.conv = nn.Conv2d(C, self.num_kp, 1).to(x.device)
        x = self.conv(x)  # (B, K, H, W)
        x_flat = x.view(B, self.num_kp, -1)
        softmax = torch.softmax(x_flat / self.temperature, dim=-1)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=x.device),
            torch.linspace(-1, 1, H, device=x.device),
            indexing='xy'
        )
        pos_x = pos_x.flatten()
        pos_y = pos_y.flatten()

        exp_x = torch.sum(softmax * pos_x, dim=-1)
        exp_y = torch.sum(softmax * pos_y, dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


class SpatialCoordinates(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, W, device=x.device),
            torch.linspace(-1.0, 1.0, H, device=x.device),
            indexing='xy'
        )
        coords = torch.stack([pos_x, pos_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class AttentionPool2d(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + 49, in_dim))  # assuming 7x7 input
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        pooled = x.mean(dim=1, keepdim=True)
        x = torch.cat([pooled, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        attn_out, _ = self.attn(x[:, :1], x, x)
        return attn_out.squeeze(1)


class ResNet(nn.Module):
    def __init__(self, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock, num_filters=64, act='relu',
                 spatial_coordinates=False, num_kp=None, attention_pool=False, average_pool=False, use_clip_stem=False):
        super().__init__()
        self.stage_sizes = stage_sizes
        self.block_cls = block_cls
        self.num_filters = num_filters
        self.act = getattr(nn, act.capitalize())
        self.spatial_coordinates = spatial_coordinates
        self.num_kp = num_kp
        self.attention_pool = attention_pool
        self.average_pool = average_pool
        self.use_clip_stem = use_clip_stem

        in_channels = 3
        self.stem = nn.Sequential()
        if use_clip_stem:
            self.stem.add_module('conv1', nn.Conv2d(in_channels, num_filters//2, 3, 2, 1))
            self.stem.add_module('norm1', nn.GroupNorm(32, num_filters//2))
            self.stem.add_module('act1', nn.ReLU())
            self.stem.add_module('conv2', nn.Conv2d(num_filters//2, num_filters//2, 3, padding=1))
            self.stem.add_module('norm2', nn.GroupNorm(32, num_filters//2))
            self.stem.add_module('act2', nn.ReLU())
            self.stem.add_module('conv3', nn.Conv2d(num_filters//2, num_filters, 3, padding=1))
            self.stem.add_module('norm3', nn.GroupNorm(32, num_filters))
            self.stem.add_module('act3', nn.ReLU())
            self.stem.add_module('pool', nn.AvgPool2d(2))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, num_filters, 7, 2, 3),
                nn.GroupNorm(32, num_filters),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1)
            )

        layers = []
        in_channels = num_filters
        for i, n_blocks in enumerate(stage_sizes):
            out_channels = num_filters * (2**i)
            for j in range(n_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                layers.append(block_cls(in_channels, out_channels, stride=stride, activation=self.act))
                in_channels = out_channels
        self.body = nn.Sequential(*layers)

        if self.num_kp is not None:
            self.head = SpatialSoftmax(num_kp=num_kp)
        elif self.attention_pool:
            self.head = AttentionPool2d(in_channels, num_heads=in_channels // 16)
        elif self.average_pool:
            self.head = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.head = nn.Identity()

    def forward(self, obs, goal=None):
        x = obs if goal is None else torch.cat([obs, goal.expand_as(obs)], dim=1)
        x = 2 * x - 1
        if self.spatial_coordinates:
            x = SpatialCoordinates()(x)
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock, num_filters=64):
        super().__init__()
        self.stage_sizes = stage_sizes
        self.block_cls = block_cls
        self.num_filters = num_filters
        self.fc = nn.Linear(512, 512)

    def forward(self, z, obs):
        B = z.size(0)
        x = self.fc(z).view(B, 512, 1, 1)
        H, W = obs.shape[-2:]  # assume (B, C, H, W)
        x = F.interpolate(x, size=(H // 32, W // 32), mode='nearest')
        for i, num_blocks in reversed(list(enumerate(self.stage_sizes))):
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                x = self.block_cls(512, self.num_filters * 2 ** i, stride=stride, activation=nn.ReLU, transpose=True)(x)
        x = F.interpolate(x, size=(H, W), mode='nearest')
        x = nn.Conv2d(self.num_filters, obs.shape[1], 3, padding=1).to(x.device)(x)
        return x