"""
Conditional Flow Matching for Image Inpainting
Implements mask-guided sampling for inpainting tasks

Compatible with your CFM training setup (unet.py, cfm_trainer.py, train_cfm.py)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class InpaintingMaskGenerator:
    """Generate various types of masks for inpainting"""
    
    @staticmethod
    def random_bbox_mask(shape: Tuple[int, int, int, int], 
                         min_size: int = 8, 
                         max_size: int = 20) -> torch.Tensor:
        """
        Generate random bounding box masks
        
        Args:
            shape: (B, C, H, W)
            min_size: minimum box dimension
            max_size: maximum box dimension
        
        Returns:
            mask: (B, 1, H, W) where 1 = keep, 0 = inpaint
        """
        B, C, H, W = shape
        masks = torch.ones(B, 1, H, W)
        
        for i in range(B):
            # Random box size
            box_h = np.random.randint(min_size, max_size + 1)
            box_w = np.random.randint(min_size, max_size + 1)
            
            # Random position
            top = np.random.randint(0, H - box_h + 1)
            left = np.random.randint(0, W - box_w + 1)
            
            # Create mask (0 in box region)
            masks[i, :, top:top+box_h, left:left+box_w] = 0
        
        return masks
    
    @staticmethod
    def center_mask(shape: Tuple[int, int, int, int], 
                    mask_size: int = 16) -> torch.Tensor:
        """
        Generate center square mask
        
        Args:
            shape: (B, C, H, W)
            mask_size: size of center square to mask
        
        Returns:
            mask: (B, 1, H, W)
        """
        B, C, H, W = shape
        masks = torch.ones(B, 1, H, W)
        
        # Center coordinates
        center_h, center_w = H // 2, W // 2
        half_size = mask_size // 2
        
        top = center_h - half_size
        left = center_w - half_size
        
        masks[:, :, top:top+mask_size, left:left+mask_size] = 0
        
        return masks
    
    @staticmethod
    def random_irregular_mask(shape: Tuple[int, int, int, int],
                             num_strokes: int = 5,
                             max_angle: float = 4,
                             max_len: int = 20,
                             max_width: int = 10) -> torch.Tensor:
        """
        Generate random irregular brush stroke masks
        
        Args:
            shape: (B, C, H, W)
            num_strokes: number of brush strokes
            max_angle: maximum angle variation
            max_len: maximum stroke length
            max_width: maximum stroke width
        
        Returns:
            mask: (B, 1, H, W)
        """
        B, C, H, W = shape
        masks = torch.ones(B, 1, H, W)
        
        for i in range(B):
            mask = np.ones((H, W), dtype=np.float32)
            
            for _ in range(num_strokes):
                # Random starting point
                start_y = np.random.randint(0, H)
                start_x = np.random.randint(0, W)
                
                # Random stroke parameters
                angle = np.random.uniform(-max_angle, max_angle)
                length = np.random.randint(10, max_len)
                width = np.random.randint(3, max_width)
                
                # Draw stroke
                for j in range(length):
                    y = int(start_y + j * np.sin(angle))
                    x = int(start_x + j * np.cos(angle))
                    
                    if 0 <= y < H and 0 <= x < W:
                        # Draw circle at this point
                        for dy in range(-width//2, width//2 + 1):
                            for dx in range(-width//2, width//2 + 1):
                                if dy**2 + dx**2 <= (width//2)**2:
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < H and 0 <= nx < W:
                                        mask[ny, nx] = 0
                    
                    # Vary angle slightly
                    angle += np.random.uniform(-0.5, 0.5)
            
            masks[i, 0] = torch.from_numpy(mask)
        
        return masks
    
    @staticmethod
    def half_image_mask(shape: Tuple[int, int, int, int], 
                        direction: str = 'left') -> torch.Tensor:
        """
        Mask half of the image
        
        Args:
            shape: (B, C, H, W)
            direction: 'left', 'right', 'top', 'bottom'
        
        Returns:
            mask: (B, 1, H, W)
        """
        B, C, H, W = shape
        masks = torch.ones(B, 1, H, W)
        
        if direction == 'left':
            masks[:, :, :, :W//2] = 0
        elif direction == 'right':
            masks[:, :, :, W//2:] = 0
        elif direction == 'top':
            masks[:, :, :H//2, :] = 0
        elif direction == 'bottom':
            masks[:, :, H//2:, :] = 0
        
        return masks


def sample_cfm_inpainting(
    model,
    flow_matcher,
    image: torch.Tensor,
    mask: torch.Tensor,
    device: str = 'cuda',
    num_steps: int = 50,
    cfg_scale: float = 2.0,
    class_labels: Optional[torch.Tensor] = None,
    num_classes: int = 10,
    resample_strategy: str = 'replace',
    jump_length: int = 1,
    num_resamples: int = 1,
):
    """
    Sample from CFM for image inpainting using mask-guided generation
    
    Args:
        model: trained UNet model (from unet.py)
        flow_matcher: ConditionalFlowMatcher instance (from cfm_trainer.py)
        image: original images (B, C, H, W) in range [-1, 1]
        mask: binary mask (B, 1, H, W) where 1 = keep, 0 = inpaint
        device: device to use
        num_steps: number of ODE integration steps
        cfg_scale: classifier-free guidance scale
        class_labels: class labels for conditional generation (B,)
        num_classes: total number of classes
        resample_strategy: 'replace' or 'repaint'
        jump_length: for repaint strategy, how many steps to jump back
        num_resamples: for repaint strategy, how many times to resample
    
    Returns:
        inpainted_images: (B, C, H, W) in range [-1, 1]
    """
    model.eval()
    
    batch_size = image.shape[0]
    image = image.to(device)
    mask = mask.to(device)
    
    # Initialize: noise in masked region, original in unmasked region
    noise = flow_matcher.sample_noise(image.shape, device)
    x = mask * image + (1 - mask) * noise
    
    # Handle class labels
    if class_labels is None:
        y = torch.randint(0, num_classes, (batch_size,), device=device)
    else:
        y = class_labels.to(device)
    
    # Unconditional labels for CFG
    y_uncond = torch.full_like(y, model.num_classes)
    
    # Time step size
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        if resample_strategy == 'replace':
            # Simple replacement strategy
            for step in range(num_steps):
                t = torch.full((batch_size,), step * dt, device=device)
                
                # Classifier-free guidance
                if cfg_scale > 1.0:
                    v_cond = model(x, t, y)
                    v_uncond = model(x, t, y_uncond)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(x, t, y)
                
                # Euler step
                x = x + dt * v
                
                # Replace unmasked regions with original
                x = mask * image + (1 - mask) * x
        
        elif resample_strategy == 'repaint':
            # RePaint strategy: jump back and resample
            t_current = 0.0
            
            while t_current < 1.0:
                # Forward steps
                for _ in range(jump_length):
                    if t_current >= 1.0:
                        break
                    
                    t = torch.full((batch_size,), t_current, device=device)
                    
                    # Predict velocity with CFG
                    if cfg_scale > 1.0:
                        v_cond = model(x, t, y)
                        v_uncond = model(x, t, y_uncond)
                        v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    else:
                        v = model(x, t, y)
                    
                    # Euler step
                    x = x + dt * v
                    t_current += dt
                    
                    # Replace unmasked regions
                    x = mask * image + (1 - mask) * x
                
                # Backward jump and add noise (resampling)
                if t_current < 1.0 and num_resamples > 0:
                    # Jump back
                    t_jump_back = max(0, t_current - jump_length * dt)
                    
                    # Add noise to masked region proportional to time step
                    noise_scale = (t_current - t_jump_back) * 0.5
                    noise = flow_matcher.sample_noise(image.shape, device)
                    x = mask * image + (1 - mask) * (x + noise_scale * noise)
                    
                    t_current = t_jump_back
    
    # Final projection
    x = mask * image + (1 - mask) * x
    
    # Clamp to valid range
    x = torch.clamp(x, -1, 1)
    
    return x


def sample_cfm_inpainting_rk4(
    model,
    flow_matcher,
    image: torch.Tensor,
    mask: torch.Tensor,
    device: str = 'cuda',
    num_steps: int = 20,
    cfg_scale: float = 2.0,
    class_labels: Optional[torch.Tensor] = None,
    num_classes: int = 10,
):
    """
    Sample using RK4 integration for higher quality inpainting
    
    Args:
        model: trained UNet model
        flow_matcher: ConditionalFlowMatcher instance
        image: original images (B, C, H, W) in range [-1, 1]
        mask: binary mask (B, 1, H, W) where 1 = keep, 0 = inpaint
        device: device to use
        num_steps: number of ODE integration steps
        cfg_scale: classifier-free guidance scale
        class_labels: class labels (B,)
        num_classes: total number of classes
    
    Returns:
        inpainted_images: (B, C, H, W) in range [-1, 1]
    """
    model.eval()
    
    batch_size = image.shape[0]
    image = image.to(device)
    mask = mask.to(device)
    
    # Initialize
    noise = flow_matcher.sample_noise(image.shape, device)
    x = mask * image + (1 - mask) * noise
    
    # Handle class labels
    if class_labels is None:
        y = torch.randint(0, num_classes, (batch_size,), device=device)
    else:
        y = class_labels.to(device)
    
    y_uncond = torch.full_like(y, model.num_classes)
    dt = 1.0 / num_steps
    
    def velocity_fn(x_in, t_in):
        """Compute velocity with CFG and mask projection"""
        if cfg_scale > 1.0:
            v_cond = model(x_in, t_in, y)
            v_uncond = model(x_in, t_in, y_uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x_in, t_in, y)
        
        # Only apply velocity to masked regions
        v = (1 - mask) * v
        return v
    
    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            
            # RK4 integration
            k1 = velocity_fn(x, t)
            k2 = velocity_fn(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = velocity_fn(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = velocity_fn(x + dt * k3, t + dt)
            
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Ensure unmasked regions remain fixed
            x = mask * image + (1 - mask) * x
    
    x = torch.clamp(x, -1, 1)
    return x


def visualize_inpainting(original: torch.Tensor,
                        masked: torch.Tensor,
                        inpainted: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """
    Create visualization grid showing: original | masked | mask | inpainted
    
    Args:
        original: (B, C, H, W)
        masked: (B, C, H, W)
        inpainted: (B, C, H, W)
        mask: (B, 1, H, W)
    
    Returns:
        grid: (B, C, H, W*4) concatenated visualization
    """
    # Convert mask to 3-channel for visualization
    mask_vis = mask.repeat(1, 3, 1, 1)
    
    # Concatenate horizontally
    grid = torch.cat([original, masked, mask_vis, inpainted], dim=3)
    
    return grid


if __name__ == "__main__":
    # Test inpainting functions
    print("Testing CFM Inpainting...")
    
    # Create dummy data
    batch_size = 4
    image = torch.randn(batch_size, 3, 32, 32)
    
    # Test mask generators
    print("\n1. Testing mask generators...")
    mask_gen = InpaintingMaskGenerator()
    
    bbox_mask = mask_gen.random_bbox_mask(image.shape)
    print(f"   Random bbox mask: {bbox_mask.shape}, {bbox_mask.sum().item():.0f} kept pixels")
    
    center_mask = mask_gen.center_mask(image.shape, mask_size=16)
    print(f"   Center mask: {center_mask.shape}, {center_mask.sum().item():.0f} kept pixels")
    
    irregular_mask = mask_gen.random_irregular_mask(image.shape)
    print(f"   Irregular mask: {irregular_mask.shape}, {irregular_mask.sum().item():.0f} kept pixels")
    
    half_mask = mask_gen.half_image_mask(image.shape, direction='left')
    print(f"   Half mask: {half_mask.shape}, {half_mask.sum().item():.0f} kept pixels")
    
    # Test visualization
    print("\n2. Testing visualization...")
    masked_image = bbox_mask * image
    inpainted = torch.randn_like(image)
    vis = visualize_inpainting(image, masked_image, inpainted, bbox_mask)
    print(f"   Visualization shape: {vis.shape}")
    
    print("\n All inpainting tests passed!")