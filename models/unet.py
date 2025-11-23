"""
Tiny UNet for CIFAR-10 (32x32 images)
Target: < 1.5M parameters
Optimized for 4GB GPU memory
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
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time and class conditioning via FiLM"""
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # GroupNorm for stability
        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        
        # FiLM conditioning
        cond_dim = time_emb_dim + class_emb_dim
        self.film = nn.Linear(cond_dim, out_channels * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Apply FiLM conditioning
        film_params = self.film(emb)[:, :, None, None]
        scale, shift = torch.chunk(film_params, 2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Lightweight self-attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
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
    """Upsampling with convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class TinyUNet(nn.Module):
    """
    Tiny UNet for CIFAR-10
    Simplified architecture with proper skip connections
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(16,),
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Class embedding
        class_emb_dim = base_channels * 4
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch_in = base_channels
        resolution = 32
        
        for level, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            
            # Add residual blocks for this level
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, time_emb_dim, class_emb_dim, dropout))
                ch_in = ch_out
            self.encoder_blocks.append(blocks)
            
            # Add attention if at specified resolution
            if resolution in attention_resolutions:
                self.encoder_attns.append(AttentionBlock(ch_out))
            else:
                self.encoder_attns.append(None)
            
            # Add downsampler (except for last level)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(ch_out))
                resolution //= 2
            else:
                self.downsamplers.append(None)
        
        # Middle blocks
        self.mid_block1 = ResBlock(ch_in, ch_in, time_emb_dim, class_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch_in)
        self.mid_block2 = ResBlock(ch_in, ch_in, time_emb_dim, class_emb_dim, dropout)
        
        # Build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            ch_out = base_channels * mult
            
            # Add residual blocks for this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # First block in each level concatenates with skip
                if i == 0:
                    blocks.append(ResBlock(ch_in + ch_out, ch_out, time_emb_dim, class_emb_dim, dropout))
                else:
                    blocks.append(ResBlock(ch_in, ch_out, time_emb_dim, class_emb_dim, dropout))
                ch_in = ch_out
            self.decoder_blocks.append(blocks)
            
            # Add attention if at specified resolution
            if resolution in attention_resolutions:
                self.decoder_attns.append(AttentionBlock(ch_out))
            else:
                self.decoder_attns.append(None)
            
            # Add upsampler (except for first level in decoder)
            if level > 0:
                self.upsamplers.append(Upsample(ch_out))
                resolution *= 2
            else:
                self.upsamplers.append(None)
        
        # Final output
        self.norm_out = nn.GroupNorm(get_num_groups(ch_in), ch_in)
        self.conv_out = nn.Conv2d(ch_in, out_channels, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, t, y):
        """
            Args:
                x: (B, C, H, W) - input images
                t: (B,) - time steps in [0, 1]
                y: (B,) - class labels
        """
        # Compute embeddings
        t_emb = self.time_mlp(t)
        y_emb = self.class_emb(y)
        emb = torch.cat([t_emb, y_emb], dim=1)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skips = []
        
        # Encoder
        for blocks, attn, downsample in zip(self.encoder_blocks, self.encoder_attns, self.downsamplers):
            # Process residual blocks
            for block in blocks:
                h = block(h, emb)
            
            # Store ONE skip per level (after all blocks processed)
            skips.append(h)
            
            # Apply attention if present
            if attn is not None:
                h = attn(h)
            
            # Downsample if present
            if downsample is not None:
                h = downsample(h)
        
        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        
        # Decoder
        for blocks, attn, upsample in zip(self.decoder_blocks, self.decoder_attns, self.upsamplers):
            # Concatenate with skip connection for first block
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            # Process residual blocks
            for block in blocks:
                h = block(h, emb)
            
            # Apply attention if present
            if attn is not None:
                h = attn(h)
            
            # Upsample if present
            if upsample is not None:
                h = upsample(h)
        
        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = TinyUNet()
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)
    y = torch.randint(0, 10, (batch_size,))
    
    with torch.no_grad():
        out = model(x, t, y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape must match input shape"
    print("Model test passed!")