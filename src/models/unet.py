"""
UNet Model Implementation for DREAM Diffusion

This module contains the UNet architecture optimized for face generation
with self-attention and efficient memory usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time conditioning and group normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = rearrange(qkv, 'b (three heads c) h w -> three b heads (h w) c',
                           three=3, heads=self.num_heads).unbind(0)
        
        attn = torch.einsum('bhqc,bhkc->bhqk', q, k) * (c // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhqk,bhkc->bhqc', attn, v)
        out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', h=h, w=w)
        
        return x + self.proj(out)


class UNet(nn.Module):
    """
    UNet model for DREAM diffusion with memory optimization.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base number of channels (default: 128)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 base_channels: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        channels = base_channels
        time_dim = channels * 4
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(channels),
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResBlock(channels, channels, time_dim, dropout),
            ResBlock(channels, channels, time_dim, dropout),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        ])
        
        self.down2 = nn.ModuleList([
            ResBlock(channels, channels * 2, time_dim, dropout),
            ResBlock(channels * 2, channels * 2, time_dim, dropout),
            nn.Conv2d(channels * 2, channels * 2, 3, stride=2, padding=1)
        ])
        
        self.down3 = nn.ModuleList([
            ResBlock(channels * 2, channels * 4, time_dim, dropout),
            ResBlock(channels * 4, channels * 4, time_dim, dropout),
            AttentionBlock(channels * 4),
            nn.Conv2d(channels * 4, channels * 4, 3, stride=2, padding=1)
        ])
        
        # Middle
        self.mid = nn.ModuleList([
            ResBlock(channels * 4, channels * 4, time_dim, dropout),
            AttentionBlock(channels * 4),
            ResBlock(channels * 4, channels * 4, time_dim, dropout)
        ])
        
        # Decoder
        self.up3 = nn.ModuleList([
            nn.ConvTranspose2d(channels * 4, channels * 4, 4, stride=2, padding=1),
            ResBlock(channels * 8, channels * 4, time_dim, dropout),
            ResBlock(channels * 4, channels * 4, time_dim, dropout),
            AttentionBlock(channels * 4)
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1),
            ResBlock(channels * 4, channels * 2, time_dim, dropout),
            ResBlock(channels * 2, channels * 2, time_dim, dropout)
        ])
        
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            ResBlock(channels * 2, channels, time_dim, dropout),
            ResBlock(channels, channels, time_dim, dropout)
        ])
        
        self.norm_out = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)
        
        # Initialize output layer to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor [B, C, H, W]
            t: Time tensor [B]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        input_size = x.shape[-2:]
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        x1 = self.conv_in(x)
        
        h1 = x1
        for layer in self.down1:
            if isinstance(layer, ResBlock):
                h1 = layer(h1, t_emb)
            else:
                h1 = layer(h1)
        
        h2 = h1
        for layer in self.down2:
            if isinstance(layer, ResBlock):
                h2 = layer(h2, t_emb)
            else:
                h2 = layer(h2)
        
        h3 = h2
        for layer in self.down3:
            if isinstance(layer, ResBlock):
                h3 = layer(h3, t_emb)
            elif isinstance(layer, AttentionBlock):
                h3 = layer(h3)
            else:
                h3 = layer(h3)
        
        # Middle
        h = h3
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        h = self.up3[0](h)
        if h.shape[-2:] != h3.shape[-2:]:
            h3_resized = F.interpolate(h3, size=h.shape[-2:], mode='bilinear', align_corners=False)
        else:
            h3_resized = h3
        h = torch.cat([h, h3_resized], dim=1)
        for layer in self.up3[1:]:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        h = self.up2[0](h)
        if h.shape[-2:] != h2.shape[-2:]:
            h2_resized = F.interpolate(h2, size=h.shape[-2:], mode='bilinear', align_corners=False)
        else:
            h2_resized = h2
        h = torch.cat([h, h2_resized], dim=1)
        for layer in self.up2[1:]:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        h = self.up1[0](h)
        if h.shape[-2:] != h1.shape[-2:]:
            h1_resized = F.interpolate(h1, size=h.shape[-2:], mode='bilinear', align_corners=False)
        else:
            h1_resized = h1
        h = torch.cat([h, h1_resized], dim=1)
        for layer in self.up1[1:]:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Final size check
        if h.shape[-2:] != input_size:
            h = F.interpolate(h, size=input_size, mode='bilinear', align_corners=False)
        
        return h
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cuda') -> 'UNet':
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        model = cls(
            in_channels=config.get('in_channels', 3),
            out_channels=config.get('out_channels', 3),
            base_channels=config.get('base_channels', 128),
            dropout=config.get('dropout', 0.1)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)