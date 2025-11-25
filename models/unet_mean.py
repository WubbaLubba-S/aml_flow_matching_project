"""
TinyUNet for MeanFlow - Notebook Version
Matches the exact architecture from your Jupyter notebook

This version uses the layer names from the notebook:
- down_blocks, up_blocks (instead of encoder_blocks, decoder_blocks)
- down_samples, up_samples (instead of downsamplers, upsamplers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_num_groups(channels):
    """Get the largest valid num_groups for GroupNorm that divides channels"""
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embeddings for time steps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class MeanFlowTimeEmbedding(nn.Module):
    """Time embedding for MeanFlow with dual time variables (r, t)"""
    def __init__(self, base_dim):
        super().__init__()
        base_dim = int(base_dim)
        self.base_dim = base_dim
        
        self.t_embed = SinusoidalTimeEmbedding(base_dim)
        self.interval_embed = SinusoidalTimeEmbedding(base_dim)
        
        time_dim = int(base_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim * 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, r, t):
        interval = t - r
        t_emb = self.t_embed(t)
        interval_emb = self.interval_embed(interval)
        combined = torch.cat([t_emb, interval_emb], dim=-1)
        return self.mlp(combined)


class ResBlock(nn.Module):
    """Residual block with time and class conditioning via FiLM"""
    def __init__(self, in_channels, out_channels, emb_dim, dropout=0.1):
        super().__init__()
        
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        emb_dim = int(emb_dim)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.film = nn.Linear(emb_dim, out_channels * 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Lightweight self-attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = attn @ v
        
        h = h.transpose(-2, -1).contiguous().view(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling with convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling with interpolation + convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class TinyUNetMeanFlow(nn.Module):
    """
    TinyUNet for MeanFlow - Notebook Architecture
    
    Uses layer names: down_blocks, up_blocks, down_samples, up_samples
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=[1, 2, 2, 2],
        num_res_blocks=2,
        attention_resolutions=[16],
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        
        base_channels = int(base_channels)
        num_classes = int(num_classes)
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        
        self.num_classes = num_classes
        
        # Time dimension
        time_dim = int(base_channels * 4)
        self.time_mlp = MeanFlowTimeEmbedding(base_channels)
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Calculate channels at each resolution
        channels = [int(base_channels * m) for m in channel_mults]
        
        # Downsampling (notebook uses down_blocks)
        self.down_blocks = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])
        
        in_ch = base_channels
        h_res = 32
        
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, time_dim, dropout))
                in_ch = out_ch
                
                if h_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            self.down_blocks.append(blocks)
            
            if i < len(channels) - 1:
                self.down_samples.append(Downsample(out_ch))
                h_res = h_res // 2
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle blocks
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, dropout)
        
        # Upsampling (notebook uses up_blocks)
        self.up_blocks = nn.ModuleList([])
        self.up_samples = nn.ModuleList([])
        
        for i, out_ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList([])
            
            skip_ch = channels[len(channels) - i - 1]
            blocks.append(ResBlock(in_ch + skip_ch, out_ch, time_dim, dropout))
            
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_ch, out_ch, time_dim, dropout))
                
                if h_res in attention_resolutions:
                    blocks.append(AttentionBlock(out_ch))
            
            self.up_blocks.append(blocks)
            
            if i < len(channels) - 1:
                self.up_samples.append(Upsample(out_ch))
                h_res = h_res * 2
                in_ch = out_ch
            else:
                self.up_samples.append(nn.Identity())
                in_ch = out_ch
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(get_num_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x, r, t, y):
        """
        Forward pass with DUAL TIME VARIABLES
        
        Args:
            x: (B, C, H, W) - input images
            r: (B,) - start time
            t: (B,) - end time
            y: (B,) - class labels
        """
        # Embed times and class
        t_emb = self.time_mlp(r, t)
        y_emb = self.class_emb(y)
        emb = t_emb + y_emb
        
        # Initial conv
        h = self.conv_in(x)
        
        # Store skip connections
        skips = []
        
        # Downsample
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                else:
                    h = block(h)
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        
        # Upsample
        for blocks, upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                else:
                    h = block(h)
            
            h = upsample(h)
        
        # Output
        h = self.conv_out(h)
        return h


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyUNetMeanFlow()
    print(f"Total parameters: {count_parameters(model):,}")
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    r = torch.rand(batch_size) * 0.5
    t = r + torch.rand(batch_size) * 0.5
    y = torch.randint(0, 10, (batch_size,))
    
    with torch.no_grad():
        out = model(x, r, t, y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Model test passed!")