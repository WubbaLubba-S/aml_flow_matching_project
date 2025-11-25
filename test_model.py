"""
Quick test script to verify model architecture and training loop
Run this before starting full training
"""

import torch
import sys
sys.path.append('.')

from models.unet import TinyUNet, count_parameters
from trainer.cfm_trainer import ConditionalFlowMatcher, sample_cfm
from trainer.ddpm_trainer import DDPMNoiseSchedule, DDPM, sample_ddpm

print("=" * 60)
print("Testing Tiny UNet Architecture")
print("=" * 60)

# Initialize model
model = TinyUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    channel_mults=[1, 2, 2],
    num_res_blocks=2,
    attention_resolutions=[16],
    num_classes=10,
    dropout=0.1,
)

num_params = count_parameters(model)
print(f"✓ Model initialized successfully")
print(f"✓ Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")

if num_params > 1.5e6:
    print(f"⚠ Warning: Model exceeds 1.5M parameter target")
else:
    print(f"✓ Model size within target (<1.5M parameters)")

# Test forward pass
print("\n" + "=" * 60)
print("Testing Forward Pass")
print("=" * 60)

batch_size = 4
x = torch.randn(batch_size, 3, 32, 32)
t = torch.rand(batch_size)
y = torch.randint(0, 10, (batch_size,))

try:
    with torch.no_grad():
        out = model(x, t, y)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test CFM loss computation
print("\n" + "=" * 60)
print("Testing Flow Matching Loss")
print("=" * 60)

flow_matcher = ConditionalFlowMatcher()

try:
    loss = flow_matcher.compute_loss(model, x, y, cfg_dropout_prob=0.1)
    print(f"✓ CFM loss computation successful")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"✓ Backward pass successful")
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    if has_grads:
        print(f"✓ Gradients computed successfully")
    else:
        print(f"✗ No gradients found")
except Exception as e:
    print(f"✗ CFM loss failed: {e}")
    sys.exit(1)

# Test CFM sampling
print("\n" + "=" * 60)
print("Testing Flow Matching Sampling")
print("=" * 60)

model.zero_grad()

try:
    samples = sample_cfm(
        model, flow_matcher, num_samples=4, num_classes=10,
        device='cpu', num_steps=10, cfg_scale=2.0
    )
    print(f"✓ CFM sampling successful")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Value range: [{samples.min():.2f}, {samples.max():.2f}]")
except Exception as e:
    print(f"✗ CFM sampling failed: {e}")
    sys.exit(1)

# Test DDPM
print("\n" + "=" * 60)
print("Testing DDPM")
print("=" * 60)

# Reinitialize model for DDPM test
model = TinyUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    channel_mults=[1, 2, 2],
    num_res_blocks=2,
    attention_resolutions=[16],
    num_classes=10,
    dropout=0.1,
)

noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
ddpm = DDPM(noise_schedule)

try:
    loss = ddpm.compute_loss(model, x, y, cfg_dropout_prob=0.1)
    print(f"✓ DDPM loss computation successful")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print(f"✓ DDPM backward pass successful")
except Exception as e:
    print(f"✗ DDPM loss failed: {e}")
    sys.exit(1)

# Test DDPM sampling
print("\n" + "=" * 60)
print("Testing DDPM Sampling")
print("=" * 60)

model.zero_grad()

try:
    samples = sample_ddpm(
        model, noise_schedule, num_samples=4, num_classes=10,
        device='cpu', num_steps=10, cfg_scale=2.0
    )
    print(f"✓ DDPM sampling successful")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Value range: [{samples.min():.2f}, {samples.max():.2f}]")
except Exception as e:
    print(f"✗ DDPM sampling failed: {e}")
    sys.exit(1)

# GPU memory test (if available)
if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("Testing GPU Memory Usage")
    print("=" * 60)
    
    device = torch.device('cuda')
    model = model.to(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    
    # Test training step with mixed precision
    x_gpu = torch.randn(16, 3, 32, 32, device=device)
    y_gpu = torch.randint(0, 10, (16,), device=device)
    
    try:
        with torch.amp.autocast('cuda', enabled=True):
            loss = flow_matcher.compute_loss(model, x_gpu, y_gpu)
        
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"✓ GPU test successful")
        print(f"  Peak memory (batch_size=16): {peak_memory:.2f} GB")
        
        if peak_memory < 4.0:
            print(f"✓ Memory usage within 4GB limit")
        else:
            print(f"⚠ Memory exceeds 4GB - reduce batch size")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\n⚠ CUDA not available - skipping GPU tests")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nYou can now start training:")
print("  python train_cfm.py --batch_size 16 --use_amp")
print("  python train_ddpm.py --batch_size 16 --use_amp")