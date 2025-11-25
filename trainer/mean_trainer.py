"""
MeanFlow Matching Training - Following Paper arXiv:2505.13447
Implements true MeanFlow with JVP and 1-step sampling

KEY DIFFERENCES FROM STANDARD FLOW MATCHING:
1. Model predicts u(z, r, t) - average velocity over [r, t]
2. Uses JVP (Jacobian-Vector Product) for time derivative
3. MeanFlow Identity: u_tgt = v_t - (t-r) * du_dt
4. 1-step sampling: x = ε - u(ε, 0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Import JVP for MeanFlow Identity computation
from torch.func import jvp as torch_jvp


def sample_time_logit_normal(batch_size, device, mu=-0.4, sigma=1.0):
    """
    Sample time from logit-normal distribution.
    Paper uses lognorm(-0.4, 1.0) as default (Table 4).
    """
    # Sample from normal distribution
    normal_sample = torch.randn(batch_size, device=device) * sigma + mu
    # Apply logistic function to map to (0, 1)
    time_sample = torch.sigmoid(normal_sample)
    return time_sample


def sample_time_pair(batch_size, device, ratio_equal=0.25, mu=-0.4, sigma=1.0):
    """
    Sample (r, t) pair following paper's methodology.

    ===== KEY DIFFERENCE: Samples TWO time variables =====
    Standard Flow Matching: Samples only t
    MeanFlow: Samples (r, t) pair with r <= t

    Args:
        batch_size: Number of samples
        device: torch device
        ratio_equal: Portion of samples where r = t (paper uses 25%)
        mu, sigma: Parameters for logit-normal distribution

    Returns:
        r, t: Time pair with r <= t
    """
    # Sample two independent times
    t1 = sample_time_logit_normal(batch_size, device, mu, sigma)
    t2 = sample_time_logit_normal(batch_size, device, mu, sigma)

    # Ensure r < t by sorting
    r = torch.minimum(t1, t2)
    t = torch.maximum(t1, t2)

    # Set ratio_equal portion to have r = t
    # Paper uses 25% (Table 1a)
    num_equal = int(batch_size * ratio_equal)
    if num_equal > 0:
        r[:num_equal] = t[:num_equal]

    return r, t


def compute_jvp_for_meanflow(model, z, r, t, y, v_t):
    """
    Compute JVP for MeanFlow training following Equation 8 in the paper.

    ===== THIS IS THE CORE OF MEANFLOW =====
    Standard Flow Matching: No JVP needed
    MeanFlow: JVP computes time derivative for MeanFlow Identity

    The time derivative is (Equation 8):
        d/dt u(z_t, r, t) = v(z_t, t) * ∂_z u + ∂_t u

    This is computed as a Jacobian-Vector Product (JVP) with tangent (v_t, 0, 1).

    Args:
        model: Network u_θ(z, r, t, y)
        z: Current state (B, C, H, W)
        r: Start time (B,)
        t: End time (B,)
        y: Class labels (B,)
        v_t: Conditional velocity (B, C, H, W)

    Returns:
        u_output: Model output u_θ(z, r, t, y)
        du_dt: Time derivative v_t * ∂_z u + ∂_t u
    """
    # Prepare function that takes (z, r, t) as inputs
    def model_fn(z_in, r_in, t_in):
        return model(z_in, r_in, t_in, y)

    # JVP tangent: (v_t, 0, 1) means:
    # - Change in z direction: v_t (the conditional velocity)
    # - Change in r direction: 0 (r is independent variable)
    # - Change in t direction: 1 (for computing d/dt)
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))

    # Compute JVP using PyTorch's functional API
    u_output, du_dt = torch_jvp(model_fn, (z, r, t), tangents)

    return u_output, du_dt


class MeanFlowMatcher:
    """
    MeanFlow Matcher - Models average velocity u(z, r, t)

    ===== KEY DIFFERENCE FROM STANDARD FLOW MATCHING =====

    Standard Flow Matching:
        - Target: v_t = ε - x
        - Loss: MSE(v_θ(z_t, t), v_t)

    MeanFlow (this implementation):
        - Target: u_tgt = v_t - (t-r) * (v_t * ∂_z u_θ + ∂_t u_θ)
        - Loss: MSE(u_θ(z_t, r, t), u_tgt)

    Paper: Equation 11, Algorithm 1
    """

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def sample_noise(self, shape, device):
        """Sample Gaussian noise"""
        return torch.randn(shape, device=device)

    def interpolate(self, x0, x1, t):
        """
        Linear interpolation: z_t = (1-t) * x_1 + t * x_0
        Paper convention: z_t = (1-t)*x + t*ε
        """
        t = t.view(-1, 1, 1, 1)
        return (1 - t) * x1 + t * x0

    def compute_loss(self, model, x1, y, cfg_dropout_prob=0.1,
                     ratio_equal=0.25, loss_p=1.0):
        """
        Compute MeanFlow loss following Algorithm 1 in paper.

        ===== THIS IS ALGORITHM 1 FROM THE PAPER =====

        Pseudocode from paper:
            t, r = sample_t_r()
            e = randn_like(x)
            z = (1 - t) * x + t * e
            v = e - x
            u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
            u_tgt = v - (t - r) * dudt
            error = u - stopgrad(u_tgt)
            loss = metric(error)

        Args:
            model: u_θ(z, r, t, y) - note 4 arguments!
            x1: Real images (B, C, H, W)
            y: Class labels (B,)
            cfg_dropout_prob: CFG dropout (paper uses 10%)
            ratio_equal: Ratio of r=t samples (paper uses 25%)
            loss_p: Adaptive weighting power (paper uses p=1.0)

        Returns:
            loss: MeanFlow loss
        """
        batch_size = x1.shape[0]
        device = x1.device

        # ===== STEP 1: Sample (r, t) pair =====
        # KEY DIFFERENCE: Standard Flow Matching samples only t
        r, t = sample_time_pair(batch_size, device, ratio_equal)

        # ===== STEP 2: Sample noise =====
        x0 = self.sample_noise(x1.shape, device)  # ε in paper

        # ===== STEP 3: Conditional velocity =====
        # v_t = ε - x (paper's default, Equation 11)
        v_t = x0 - x1

        # ===== STEP 4: Classifier-free guidance dropout =====
        if cfg_dropout_prob > 0:
            mask = torch.rand(batch_size, device=device) < cfg_dropout_prob
            y = torch.where(mask, torch.full_like(y, model.num_classes), y)

        # ===== STEP 5: Interpolate =====
        # z_t = (1-t)*x + t*ε
        z_t = self.interpolate(x0, x1, t)

        # ===== STEP 6: MEANFLOW IDENTITY COMPUTATION =====
        # This is THE KEY part that differs from standard Flow Matching

        # Compute u_θ and its time derivative using JVP
        u_pred, du_dt = compute_jvp_for_meanflow(model, z_t, r, t, y, v_t)

        # MeanFlow target (Equation 11)
        # u_tgt = v_t - (t - r) * du_dt
        u_tgt = v_t - (t - r).view(-1, 1, 1, 1) * du_dt

        # ===== STEP 7: Loss with stop-gradient on target =====
        error = u_pred - u_tgt.detach()  # stop-gradient on target!

        # ===== STEP 8: Adaptive loss weighting =====
        # Paper Section 4.3, Equation 22
        # w = 1 / (||error||^2 + c)^p
        if loss_p != 0:
            error_norm = (error ** 2).mean(dim=[1, 2, 3], keepdim=True)
            weight = 1.0 / (error_norm + 1e-3) ** loss_p
            loss = (weight.detach() * (error ** 2)).mean()
        else:
            # Standard L2 loss
            loss = F.mse_loss(error, torch.zeros_like(error))

        return loss


class MeanFlowTrainer:
    """Training loop for MeanFlow"""
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
    
    def train_epoch(self, dataloader, epoch, cfg_dropout_prob=0.1,
                    ratio_equal=0.25, loss_p=1.0):
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
                    self.model, images, labels, 
                    cfg_dropout_prob=cfg_dropout_prob,
                    ratio_equal=ratio_equal,
                    loss_p=loss_p
                )
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping (paper uses max_norm=1.0)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Logging
            with torch.no_grad():
                total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Clear cache at end of epoch
        torch.cuda.empty_cache()
        
        return total_loss / num_batches


def sample_meanflow_1step(
    model,
    num_samples,
    num_classes,
    device='cuda',
    cfg_scale=2.0,
    class_labels=None,
):
    """
    1-step sampling for MeanFlow (Algorithm 2 from paper)

    ===== THIS IS ALGORITHM 2 FROM THE PAPER =====

    Pseudocode from paper:
        e = randn(x_shape)
        x = e - fn(e, r=0, t=1)

    KEY DIFFERENCE: Direct computation, NO ODE integration!

    Standard Flow Matching: Needs ODE integration over 50+ steps
    MeanFlow: Direct 1-step formula!

    Formula: x = ε - u(ε, 0, 1)

    Args:
        model: Trained MeanFlow model u_θ
        num_samples: Number of samples to generate
        num_classes: Number of classes
        device: Device
        cfg_scale: Classifier-free guidance scale
        class_labels: Optional specific class labels

    Returns:
        Generated samples (num_samples, 3, 32, 32)
    """
    model.eval()

    # ===== STEP 1: Start from noise =====
    epsilon = torch.randn(num_samples, 3, 32, 32, device=device)

    # ===== STEP 2: Sample or use provided labels =====
    if class_labels is None:
        y = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        y = class_labels.to(device)

    # ===== STEP 3: Time variables for 1-step sampling =====
    # r=0 (start), t=1 (end)
    r = torch.zeros(num_samples, device=device)
    t = torch.ones(num_samples, device=device)

    with torch.no_grad():
        if cfg_scale > 1.0:
            # ===== With CFG =====
            y_uncond = torch.full_like(y, model.num_classes)

            u_cond = model(epsilon, r, t, y)
            u_uncond = model(epsilon, r, t, y_uncond)

            # CFG: u = u_uncond + scale * (u_cond - u_uncond)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            # ===== Without CFG =====
            u = model(epsilon, r, t, y)

        # ===== STEP 4: 1-step sampling =====
        # x = ε - u(ε, 0, 1)
        # This is it! No ODE integration needed!
        x = epsilon - u

    return torch.clamp(x, -1, 1)


if __name__ == "__main__":
    # Test MeanFlow matching
    import sys
    sys.path.append('.')
    from models.unet_mean import TinyUNetMeanFlow
    
    model = TinyUNetMeanFlow()
    flow_matcher = MeanFlowMatcher()
    
    # Test loss computation
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    loss = flow_matcher.compute_loss(model, x1, y)
    print(f"MeanFlow Loss: {loss.item():.4f}")
    
    # Test 1-step sampling
    samples = sample_meanflow_1step(model, 4, 10, device='cpu')
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    print("\n✓ MeanFlow matching tests passed!")
    print("  ✓ JVP computation works")
    print("  ✓ Loss computation works")
    print("  ✓ 1-step sampling works")