"""
DDPM Training with Cosine Noise Schedule
CORRECTED VERSION - Fixed sampling function to match notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math


class DDPMNoiseSchedule:
    """
    Cosine noise schedule for DDPM
    
    Based on "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
    """
    def __init__(self, num_timesteps=1000, s=0.008):
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute alpha schedule using cosine
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Store as buffers (not parameters)
        self.alphas_cumprod = alphas_cumprod[:-1]  # 200 values (for timesteps 0-199)
        
        # Previous alphas: prepend 1.0, then take first (num_timesteps-1) values
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), 
            self.alphas_cumprod[:-1]
        ])  # Result: 1 + 199 = 200 values
        
        # Pre-compute values for training and sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
            * (1.0 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        
    def add_noise(self, x0, t, noise):
        """
        Forward diffusion: q(x_t | x_0)
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise


class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, noise_schedule):
        self.noise_schedule = noise_schedule
        
    def compute_loss(self, model, x0, y, cfg_dropout_prob=0.1):
        """
        Compute DDPM training loss (noise prediction)
        
        Args:
            model: UNet model that predicts noise
            x0: clean images (B, C, H, W)
            y: class labels (B,)
            cfg_dropout_prob: probability of unconditional training
        
        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Classifier-free guidance: randomly drop conditioning
        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, device=device) < cfg_dropout_prob
            y = torch.where(mask, torch.full_like(y, model.num_classes), y)
        
        # Add noise to images
        x_t = self.noise_schedule.add_noise(x0, t, noise)
        
        # Normalize timesteps to [0, 1] for model input (matching CFM interface)
        t_normalized = t.float() / self.noise_schedule.num_timesteps
        
        # Predict noise
        noise_pred = model(x_t, t_normalized, y)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss


class DDPMTrainer:
    """Training loop for DDPM"""
    def __init__(
        self,
        model,
        optimizer,
        ddpm,
        device='cuda',
        use_amp=True,
        grad_accum_steps=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.ddpm = ddpm
        self.device = device
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Move model and noise schedule to device
        self.model.to(device)
        self.ddpm.noise_schedule.alphas_cumprod = self.ddpm.noise_schedule.alphas_cumprod.to(device)
        self.ddpm.noise_schedule.alphas_cumprod_prev = self.ddpm.noise_schedule.alphas_cumprod_prev.to(device)
        self.ddpm.noise_schedule.sqrt_alphas_cumprod = self.ddpm.noise_schedule.sqrt_alphas_cumprod.to(device)
        self.ddpm.noise_schedule.sqrt_one_minus_alphas_cumprod = self.ddpm.noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
        self.ddpm.noise_schedule.posterior_variance = self.ddpm.noise_schedule.posterior_variance.to(device)
    
    def train_epoch(self, dataloader, epoch, cfg_dropout_prob=0.1):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                loss = self.ddpm.compute_loss(self.model, images, labels, cfg_dropout_prob)
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
                self.optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        return total_loss / num_batches


def sample_ddpm(
    model,
    noise_schedule,
    num_samples,
    num_classes,
    device='cuda',
    num_steps=100,
    cfg_scale=2.0,
    class_labels=None,
    eta=0.0,
    show_progress=False,
):
    """
    Sample from DDPM using DDIM sampling (CORRECTED VERSION)
    
    Key fixes from notebook:
    1. Proper integer timestep indexing
    2. Correct alpha_bar_t_prev handling  
    3. Simplified DDIM formula
    
    Args:
        model: trained UNet
        noise_schedule: DDPMNoiseSchedule instance
        num_samples: number of images to generate
        num_classes: number of classes
        device: device to use
        num_steps: number of denoising steps (can be < num_timesteps)
        cfg_scale: classifier-free guidance scale
        class_labels: specific class labels (optional)
        eta: DDIM interpolation parameter (0=deterministic, 1=DDPM)
        show_progress: whether to show progress bar (default: False)
    
    Returns:
        samples: (num_samples, 3, 32, 32) generated images
    """
    model.eval()
    
    # Start from pure noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    
    # Sample or use provided class labels
    if class_labels is None:
        y = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        y = class_labels.to(device)
    
    # Unconditional labels for CFG
    y_uncond = torch.full_like(y, model.num_classes)
    
    # Create timestep sequence: [199, 197, ..., 2, 0] for 100 steps with 200 timesteps
    timesteps = torch.linspace(
        noise_schedule.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device
    )
    
    # Use iterator with optional progress bar
    iterator = tqdm(timesteps, desc="DDPM Sampling", leave=False) if show_progress else timesteps
    
    with torch.no_grad():
        for i, t in enumerate(iterator):
            # Current timestep for all samples in batch
            t_batch = t.repeat(num_samples)
            
            # CRITICAL: Normalize to [0, 1] for model (matching training)
            t_normalized = t_batch.float() / noise_schedule.num_timesteps
            
            # Predict noise with classifier-free guidance
            if cfg_scale > 1.0:
                noise_cond = model(x, t_normalized, y)
                noise_uncond = model(x, t_normalized, y_uncond)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model(x, t_normalized, y)
            
            # Get schedule values - FIXED: use integer index
            t_int = int(t.item())
            
            # Current alpha_bar
            alpha_bar_t = noise_schedule.alphas_cumprod[t_int]
            
            # Previous alpha_bar - FIXED: handle boundary correctly
            if t_int > 0:
                alpha_bar_t_prev = noise_schedule.alphas_cumprod[t_int - 1]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)
            
            # Compute helper variables
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
            sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1.0 - alpha_bar_t_prev)
            
            # Predict x0 from x_t and noise prediction
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # DDIM update (deterministic when eta=0)
            if i < len(timesteps) - 1:
                # DDIM formula: x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1}) * noise_pred
                x = sqrt_alpha_bar_t_prev * pred_x0 + sqrt_one_minus_alpha_bar_t_prev * noise_pred
                
                # Add stochastic noise if eta > 0 (full DDPM sampling)
                if eta > 0:
                    # Variance for DDPM sampling
                    sigma = eta * torch.sqrt(
                        (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * 
                        (1 - alpha_bar_t / alpha_bar_t_prev)
                    )
                    noise = torch.randn_like(x)
                    x = x + sigma * noise
            else:
                # Final step: return predicted x0
                x = pred_x0
    
    # Clamp final output
    x = torch.clamp(x, -1, 1)
    return x


if __name__ == "__main__":
    # Test DDPM
    from models.unet import TinyUNet
    
    model = TinyUNet()
    noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
    ddpm = DDPM(noise_schedule)
    
    # Test loss computation
    batch_size = 4
    x0 = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    loss = ddpm.compute_loss(model, x0, y)
    print(f"DDPM Loss: {loss.item():.4f}")
    
    # Test sampling
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
    noise_schedule.alphas_cumprod_prev = noise_schedule.alphas_cumprod_prev.to(device)
    noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
    noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
    noise_schedule.posterior_variance = noise_schedule.posterior_variance.to(device)
    
    samples = sample_ddpm(model, noise_schedule, 4, 10, device=device, num_steps=10)
    print(f"Sample shape: {samples.shape}")
    print("DDPM tests passed!")