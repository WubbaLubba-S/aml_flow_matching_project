"""
Conditional Flow Matching Training
Implements Rectified Flow (linear interpolation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import sys

class ConditionalFlowMatcher:
    """
    Conditional Flow Matching with Rectified Flow
    
    Uses linear interpolation: x_t = t * x_1 + (1-t) * x_0
    Target velocity: v_t = x_1 - x_0
    """
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min  # Small noise for numerical stability
        
    def sample_time(self, batch_size, device):
        """Sample random time steps uniformly from [0, 1]"""
        return torch.rand(batch_size, device=device)
    
    def sample_noise(self, shape, device):
        """Sample Gaussian noise"""
        return torch.randn(shape, device=device)
    
    def interpolate(self, x0, x1, t):
        """
        Linear interpolation path: x_t = t * x_1 + (1-t) * x_0
        
        Args:
            x0: noise (B, C, H, W)
            x1: data (B, C, H, W)
            t: time (B,) in [0, 1]
        
        Returns:
            x_t: interpolated samples
        """
        t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        return t * x1 + (1 - t) * x0
    
    def target_velocity(self, x0, x1):
        """
        Target velocity field for rectified flow: v_t = x_1 - x_0
        
        This is the derivative of the linear interpolation path.
        """
        return x1 - x0
    
    def compute_loss(self, model, x1, y, cfg_dropout_prob=0.1):
        """
        Compute flow matching loss
        
        Args:
            model: UNet model that predicts velocity
            x1: real images (B, C, H, W)
            y: class labels (B,)
            cfg_dropout_prob: probability of unconditional training for CFG
        
        Returns:
            loss: MSE between predicted and target velocity
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample noise and time
        x0 = self.sample_noise(x1.shape, device)
        t = self.sample_time(batch_size, device)
        
        # Classifier-free guidance: randomly drop class conditioning
        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, device=device) < cfg_dropout_prob
            y = torch.where(mask, torch.full_like(y, model.num_classes), y)
        
        # Interpolate to get x_t
        x_t = self.interpolate(x0, x1, t)
        
        # Target velocity
        v_target = self.target_velocity(x0, x1)
        
        # Predict velocity
        v_pred = model(x_t, t, y)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss


class CFMTrainer:
    """Training loop for Conditional Flow Matching"""
    def __init__(
        self,
        model,
        optimizer,
        flow_matcher,
        device='cuda',
        use_amp=True,
        grad_accum_steps=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.device = device
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self, dataloader, epoch, cfg_dropout_prob=0.1):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Clear cache at start of epoch
        torch.cuda.empty_cache()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                loss = self.flow_matcher.compute_loss(
                    self.model, images, labels, cfg_dropout_prob
                )
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  
            
            # Logging - ensure we detach from graph
            with torch.no_grad():
                total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})
            
            #  Clear cache every 50 batches to prevent fragmentation
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        #  Clear cache at end of epoch
        torch.cuda.empty_cache()
        
        return total_loss / num_batches


def sample_cfm(
    model,
    flow_matcher,
    num_samples,
    num_classes,
    device='cuda',
    num_steps=50,
    cfg_scale=2.0,
    class_labels=None,
):
    """
    Sample from the flow matching model using Euler ODE solver
    
    Args:
        model: trained UNet
        flow_matcher: ConditionalFlowMatcher instance
        num_samples: number of images to generate
        num_classes: number of classes
        device: device to use
        num_steps: number of ODE integration steps
        cfg_scale: classifier-free guidance scale
        class_labels: specific class labels (if None, sample uniformly)
    
    Returns:
        samples: (num_samples, 3, 32, 32)
    """
    model.eval()
    
    # Sample initial noise
    x = flow_matcher.sample_noise((num_samples, 3, 32, 32), device)
    
    # Sample or use provided class labels
    if class_labels is None:
        y = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        y = class_labels.to(device)
    
    # Unconditional labels for CFG
    y_uncond = torch.full_like(y, model.num_classes)
    
    # Time steps for ODE integration
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full((num_samples,), step * dt, device=device)
            
            # Classifier-free guidance
            if cfg_scale > 1.0:
                # Conditional prediction
                v_cond = model(x, t, y)
                # Unconditional prediction
                v_uncond = model(x, t, y_uncond)
                # Combine with guidance scale
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, y)
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * v
    
    # Clamp to valid image range
    x = torch.clamp(x, -1, 1)
    
    return x


def sample_cfm_rk4(
    model,
    flow_matcher,
    num_samples,
    num_classes,
    device='cuda',
    num_steps=20,
    cfg_scale=2.0,
    class_labels=None,
):
    """
    Sample using RK4 (Runge-Kutta 4th order) for higher quality with fewer steps
    """
    model.eval()
    
    x = flow_matcher.sample_noise((num_samples, 3, 32, 32), device)
    
    if class_labels is None:
        y = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        y = class_labels.to(device)
    
    y_uncond = torch.full_like(y, model.num_classes)
    dt = 1.0 / num_steps
    
    def velocity_fn(x_in, t_in):
        """Compute velocity with CFG"""
        if cfg_scale > 1.0:
            v_cond = model(x_in, t_in, y)
            v_uncond = model(x_in, t_in, y_uncond)
            return v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            return model(x_in, t_in, y)
    
    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full((num_samples,), step * dt, device=device)
            
            # RK4 integration
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    x = torch.clamp(x, -1, 1)
    return x


if __name__ == "__main__":
    # Test flow matching
    current_dir = os.path.dirname(os.path.abspath(__file__))

#  Get the parent folder (The root of your project)
    parent_dir = os.path.dirname(current_dir)

    #  Add paths so Python can see everything
    sys.path.append(current_dir) 
    sys.path.append(parent_dir)  
    from models.unet import TinyUNet
    
    model = TinyUNet()
    flow_matcher = ConditionalFlowMatcher()
    
    # Test loss computation
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    loss = flow_matcher.compute_loss(model, x1, y)
    print(f"CFM Loss: {loss.item():.4f}")
    
    # Test sampling
    samples = sample_cfm(model, flow_matcher, 4, 10, device='cpu', num_steps=10)
    print(f"Sample shape: {samples.shape}")
    print("CFM tests passed!")