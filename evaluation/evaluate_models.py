"""
Comprehensive Evaluation Script (FIXED VERSION)
- FID score calculation (overall and per-class)
- KID (Kernel Inception Distance) calculation (overall and per-class)
- Inference speed comparison
- Memory usage analysis
- Qualitative sample grids

FIXES APPLIED:
1. Sampler classes to avoid repeated initialization
2. Memory management with cache clearing
3. Fixed default base_channels (28 instead of 32)
4. Better progress tracking
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import json
import scipy
import os
# Import models
import sys
sys.path.append('.')

current_dir = os.path.dirname(os.path.abspath(__file__))

#  Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

#  Add paths so Python can see everything
sys.path.append(current_dir) 
sys.path.append(parent_dir)  


from models.unet import TinyUNet
from trainer.cfm_trainer import ConditionalFlowMatcher, sample_cfm, sample_cfm_rk4
from trainer.ddpm_trainer import DDPMNoiseSchedule, DDPM, sample_ddpm

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class CFMSampler:
    """Efficient CFM sampling with reusable flow matcher"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.flow_matcher = ConditionalFlowMatcher()
        print("✓ CFM sampler initialized")
    
    def sample(self, num_samples, num_steps=50, cfg_scale=2.0, class_labels=None):
        return sample_cfm(
            self.model, self.flow_matcher, num_samples, 10,
            device=self.device, num_steps=num_steps,
            cfg_scale=cfg_scale, class_labels=class_labels
        )


class DDPMSampler:
    """Efficient DDPM sampling with reusable noise schedule"""
    def __init__(self, model, device='cuda', num_timesteps=200):
        self.model = model
        self.device = device
        print(f"Initializing DDPM noise schedule (num_timesteps={num_timesteps})...")
        self.noise_schedule = DDPMNoiseSchedule(num_timesteps=num_timesteps)
        
        # Move to device ONCE
        self.noise_schedule.alphas_cumprod = self.noise_schedule.alphas_cumprod.to(device)
        self.noise_schedule.alphas_cumprod_prev = self.noise_schedule.alphas_cumprod_prev.to(device)
        self.noise_schedule.sqrt_alphas_cumprod = self.noise_schedule.sqrt_alphas_cumprod.to(device)
        self.noise_schedule.sqrt_one_minus_alphas_cumprod = self.noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
        self.noise_schedule.posterior_variance = self.noise_schedule.posterior_variance.to(device)
        print("✓ DDPM sampler initialized")
    
    def sample(self, num_samples, num_steps=100, cfg_scale=2.0, class_labels=None, show_progress=False):
        """Sample with optional progress bar control"""
        return sample_ddpm(
            self.model, self.noise_schedule, num_samples, 10,
            device=self.device, num_steps=num_steps,
            cfg_scale=cfg_scale, class_labels=class_labels, show_progress=show_progress
        )


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint (handle different formats)
    if 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            config = args
        else:
            # args is a Namespace object, convert to dict
            config = vars(args) if hasattr(args, '__dict__') else {}
    else:
        # No args in checkpoint, use defaults
        print("Warning: No 'args' found in checkpoint, using default configuration")
        config = {}
    
    # Initialize model with configuration
    model = TinyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=config.get('base_channels', 32),  # Fixed: was 32
        channel_mults=config.get('channel_mults', [1, 2, 2]),
        num_res_blocks=config.get('num_res_blocks', 2),
        attention_resolutions=config.get('attention_resolutions', [16]),
        num_classes=10,
        dropout=config.get('dropout', 0.1),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print checkpoint info
    print(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.4f}")
    elif 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return model, checkpoint


def compute_inception_features(images, inception_model, device='cuda', batch_size=50):
    """Extract Inception features from images"""
    inception_model.eval()
    features_list = []
    
    # Resize images to 299x299 for InceptionV3
    resize = transforms.Resize((299, 299), antialias=True)
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing features", leave=False):
            batch = images[i * batch_size : (i + 1) * batch_size].to(device)
            
            # Resize and normalize for Inception
            batch_resized = resize(batch)
            
            # Get features
            features = inception_model(batch_resized)
            features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    
    return features


def compute_fid_statistics(images, inception_model, device='cuda', batch_size=50):
    """Compute mean and covariance of Inception features for FID"""
    features = compute_inception_features(images, inception_model, device, batch_size)
    
    # Compute statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two Gaussian distributions"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid


def polynomial_kernel(X, Y, degree=3, gamma=None, coef=1.0):
    """Compute polynomial kernel between two sets of features"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    kernel = (gamma * np.dot(X, Y.T) + coef) ** degree
    return kernel


def calculate_mmd(features_real, features_fake, kernel_fn, subset_size=1000):
    """Calculate Maximum Mean Discrepancy (MMD) between real and fake features"""
    n = min(len(features_real), subset_size)
    m = min(len(features_fake), subset_size)
    
    # Randomly sample subsets
    idx_real = np.random.choice(len(features_real), n, replace=False)
    idx_fake = np.random.choice(len(features_fake), m, replace=False)
    
    X = features_real[idx_real]
    Y = features_fake[idx_fake]
    
    # Compute kernel matrices
    K_XX = kernel_fn(X, X)
    K_YY = kernel_fn(Y, Y)
    K_XY = kernel_fn(X, Y)
    
    # Unbiased estimators
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    term1 = K_XX.sum() / (n * (n - 1))
    term2 = K_YY.sum() / (m * (m - 1))
    term3 = K_XY.sum() / (n * m)
    
    mmd_squared = term1 + term2 - 2 * term3
    
    return mmd_squared


def calculate_kid(features_real, features_fake, num_subsets=100, subset_size=1000, 
                  degree=3, gamma=None, coef=1.0):
    """Calculate Kernel Inception Distance (KID)"""
    if gamma is None:
        gamma = 1.0 / features_real.shape[1]
    
    # Define kernel function
    def kernel_fn(X, Y):
        return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef=coef)
    
    kid_values = []
    
    for _ in tqdm(range(num_subsets), desc="Computing KID subsets", leave=False):
        mmd_squared = calculate_mmd(features_real, features_fake, kernel_fn, subset_size)
        kid_values.append(mmd_squared)
    
    kid_values = np.array(kid_values)
    kid_mean = np.mean(kid_values)
    kid_std = np.std(kid_values)
    
    return kid_mean, kid_std


def get_inception_model(device='cuda'):
    """Load pre-trained InceptionV3 for feature extraction"""
    try:
        import scipy.linalg
        from torchvision.models import inception_v3
        
        # Load pre-trained Inception
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()  # Remove final classification layer
        inception.to(device)
        inception.eval()
        
        return inception
    except ImportError:
        print("ERROR: scipy is required for FID/KID calculation. Install with: pip install scipy")
        return None


def generate_class_samples(sampler, class_idx, num_samples, num_steps, batch_size=100):
    """Generate samples for a specific class using sampler"""
    generated_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        batch_size_curr = min(batch_size, num_samples - i * batch_size)
        
        # Create class labels for this batch
        class_labels = torch.full((batch_size_curr,), class_idx, dtype=torch.long, device=sampler.device)
        
        # Use sampler (already initialized, no repeated creation!)
        samples = sampler.sample(batch_size_curr, num_steps=num_steps, 
                                cfg_scale=2.0, class_labels=class_labels)
        
        generated_images.append(samples.cpu())
    
    return torch.cat(generated_images, dim=0)[:num_samples]


def load_class_images(dataset, class_idx, num_samples):
    """Load real images for a specific class from CIFAR-10"""
    class_images = []
    class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    
    # Sample from available class images
    if len(class_indices) < num_samples:
        # If not enough images, sample with replacement
        selected_indices = np.random.choice(class_indices, num_samples, replace=True)
    else:
        selected_indices = np.random.choice(class_indices, num_samples, replace=False)
    
    for idx in selected_indices:
        class_images.append(dataset[idx][0])
    
    return torch.stack(class_images)


def evaluate_fid_per_class(sampler, model_type, samples_per_class=1000,
                           num_steps=50, device='cuda'):
    """Evaluate FID score for each class separately"""
    print(f"\n{'='*60}")
    print(f"Evaluating per-class FID for {model_type.upper()} (num_steps={num_steps})")
    print(f"{'='*60}")
    
    # Load Inception model
    inception = get_inception_model(device)
    if inception is None:
        return None
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    fid_scores = {}
    
    for class_idx in tqdm(range(10), desc="Computing FID per class"):
        class_name = CIFAR10_CLASSES[class_idx]
        
        # Generate samples for this class
        generated_images = generate_class_samples(
            sampler, class_idx, samples_per_class, num_steps
        )
        
        # Load real images for this class
        real_images = load_class_images(dataset, class_idx, samples_per_class)
        
        # Compute statistics
        mu_gen, sigma_gen = compute_fid_statistics(generated_images, inception, device)
        mu_real, sigma_real = compute_fid_statistics(real_images, inception, device)
        
        # Calculate FID
        fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        fid_scores[class_name] = float(fid_score)
        
        print(f"  {class_name:12s}: FID = {fid_score:.2f}")
    
    # Compute average
    avg_fid = np.mean(list(fid_scores.values()))
    fid_scores['average'] = float(avg_fid)
    print(f"  {'Average':12s}: FID = {avg_fid:.2f}")
    print(f"{'='*60}\n")
    
    return fid_scores


def evaluate_kid_per_class(sampler, model_type, samples_per_class=1000,
                           num_steps=50, num_subsets=100, subset_size=500,
                           device='cuda'):
    """Evaluate KID score for each class separately"""
    print(f"\n{'='*60}")
    print(f"Evaluating per-class KID for {model_type.upper()} (num_steps={num_steps})")
    print(f"{'='*60}")
    
    # Load Inception model
    inception = get_inception_model(device)
    if inception is None:
        return None
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    kid_scores = {}
    
    for class_idx in tqdm(range(10), desc="Computing KID per class"):
        class_name = CIFAR10_CLASSES[class_idx]
        
        # Generate samples for this class
        generated_images = generate_class_samples(
            sampler, class_idx, samples_per_class, num_steps
        )
        
        # Load real images for this class
        real_images = load_class_images(dataset, class_idx, samples_per_class)
        
        # Compute features
        features_gen = compute_inception_features(generated_images, inception, device)
        features_real = compute_inception_features(real_images, inception, device)
        
        # Calculate KID
        kid_mean, kid_std = calculate_kid(
            features_real, features_gen,
            num_subsets=num_subsets,
            subset_size=subset_size,
            degree=3,
            gamma=None,
            coef=1.0
        )
        
        kid_scores[class_name] = {
            'mean': float(kid_mean),
            'std': float(kid_std)
        }
        
        print(f"  {class_name:12s}: KID = {kid_mean:.6f} ± {kid_std:.6f}")
    
    # Compute average
    avg_kid_mean = np.mean([v['mean'] for v in kid_scores.values()])
    avg_kid_std = np.mean([v['std'] for v in kid_scores.values()])
    kid_scores['average'] = {
        'mean': float(avg_kid_mean),
        'std': float(avg_kid_std)
    }
    print(f"  {'Average':12s}: KID = {avg_kid_mean:.6f} ± {avg_kid_std:.6f}")
    print(f"{'='*60}\n")
    
    return kid_scores


def evaluate_fid(sampler, model_type, num_samples=10000, batch_size=100, 
                 num_steps=50, device='cuda', balanced_classes=True):
    """Evaluate FID score"""
    print(f"\n{'='*60}")
    print(f"Evaluating overall FID for {model_type.upper()} (num_steps={num_steps})")
    print(f"{'='*60}")
    
    # Load Inception model
    inception = get_inception_model(device)
    if inception is None:
        return None
    
    # Generate samples
    if balanced_classes:
        num_classes = 10
        samples_per_class = num_samples // num_classes
        print(f"Generating {num_samples} samples ({samples_per_class} per class)...")
    else:
        print(f"Generating {num_samples} samples (random classes)...")
    
    generated_images = []
    
    if balanced_classes:
        # Generate balanced samples (equal number per class)
        for class_idx in tqdm(range(num_classes), desc="Generating per class"):
            class_samples = generate_class_samples(
                sampler, class_idx, samples_per_class, num_steps
            )
            generated_images.append(class_samples)
        
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
        
    else:
        # Original random class sampling
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            batch_size_curr = min(batch_size, num_samples - i * batch_size)
            
            # Use sampler directly
            samples = sampler.sample(batch_size_curr, num_steps=num_steps, cfg_scale=2.0)
            generated_images.append(samples.cpu())
        
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    # Load real CIFAR-10 images
    print("Loading real CIFAR-10 images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Sample random subset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_images = torch.stack([dataset[i][0] for i in tqdm(indices, desc="Loading", leave=False)])
    
    # Compute statistics
    print("Computing features for generated images...")
    mu_gen, sigma_gen = compute_fid_statistics(generated_images, inception, device)
    
    print("Computing features for real images...")
    mu_real, sigma_real = compute_fid_statistics(real_images, inception, device)
    
    # Calculate FID
    import scipy.linalg
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    
    print(f"✓ Overall FID Score: {fid_score:.2f}")
    print(f"{'='*60}\n")
    
    return fid_score


def evaluate_kid(sampler, model_type, num_samples=10000, batch_size=100,
                 num_steps=50, num_subsets=100, subset_size=1000, 
                 device='cuda', balanced_classes=True):
    """Evaluate Kernel Inception Distance (KID)"""
    print(f"\n{'='*60}")
    print(f"Evaluating overall KID for {model_type.upper()} (num_steps={num_steps})")
    print(f"{'='*60}")
    
    # Load Inception model
    inception = get_inception_model(device)
    if inception is None:
        return None, None
    
    # Generate samples
    if balanced_classes:
        num_classes = 10
        samples_per_class = num_samples // num_classes
        print(f"Generating {num_samples} samples ({samples_per_class} per class)...")
    else:
        print(f"Generating {num_samples} samples (random classes)...")
    
    generated_images = []
    
    if balanced_classes:
        # Generate balanced samples (equal number per class)
        for class_idx in tqdm(range(num_classes), desc="Generating per class"):
            class_samples = generate_class_samples(
                sampler, class_idx, samples_per_class, num_steps
            )
            generated_images.append(class_samples)
        
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
        
    else:
        # Random class sampling
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            batch_size_curr = min(batch_size, num_samples - i * batch_size)
            
            # Use sampler directly
            samples = sampler.sample(batch_size_curr, num_steps=num_steps, cfg_scale=2.0)
            generated_images.append(samples.cpu())
        
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    # Load real CIFAR-10 images
    print("Loading real CIFAR-10 images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Sample random subset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    real_images = torch.stack([dataset[i][0] for i in tqdm(indices, desc="Loading", leave=False)])
    
    # Compute features
    print("Computing features for generated images...")
    features_gen = compute_inception_features(generated_images, inception, device)
    
    print("Computing features for real images...")
    features_real = compute_inception_features(real_images, inception, device)
    
    # Calculate KID
    print(f"Computing KID with {num_subsets} subsets...")
    kid_mean, kid_std = calculate_kid(
        features_real, features_gen,
        num_subsets=num_subsets,
        subset_size=subset_size,
        degree=3,
        gamma=None,
        coef=1.0
    )
    
    print(f"✓ Overall KID Score: {kid_mean:.6f} ± {kid_std:.6f}")
    print(f"{'='*60}\n")
    
    return kid_mean, kid_std


def measure_inference_speed(sampler, model_type, num_samples=100, num_steps_list=[20, 50, 100]):
    """Measure inference speed for different numbers of sampling steps"""
    print(f"\n{'='*60}")
    print(f"Measuring inference speed for {model_type.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    for num_steps in num_steps_list:
        # Warm-up
        _ = sampler.sample(4, num_steps=num_steps)
        torch.cuda.synchronize()
        
        # Actual timing
        start_time = time.time()
        _ = sampler.sample(num_samples, num_steps=num_steps)
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_sample = total_time / num_samples
        
        results[f'steps_{num_steps}'] = {
            'total_time': total_time,
            'time_per_sample': time_per_sample,
            'samples_per_second': num_samples / total_time,
        }
        
        print(f"  Steps={num_steps:3d}: {time_per_sample:.4f}s/sample, "
              f"{num_samples/total_time:6.2f} samples/s")
    
    print(f"{'='*60}\n")
    return results


def measure_memory_usage(sampler, model_type, batch_size=32, device='cuda'):
    """Measure peak GPU memory usage during sampling"""
    print(f"\n{'='*60}")
    print(f"Measuring memory usage for {model_type.upper()}")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    # Sample with typical settings
    _ = sampler.sample(batch_size, num_steps=50)
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    print(f"  Peak memory usage: {peak_memory:.2f} GB")
    print(f"{'='*60}\n")
    
    return peak_memory


def generate_sample_grids(sampler, model_type, output_dir, num_steps=50, device='cuda'):
    """Generate qualitative sample grids"""
    print(f"\n{'='*60}")
    print(f"Generating sample grids for {model_type.upper()}")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Class-conditional samples
    print(f"Generating class-conditional samples (num_steps={num_steps})...")
    for class_idx in tqdm(range(10), desc="Generating grids"):
        class_labels = torch.full((64,), class_idx, dtype=torch.long, device=device)
        
        samples = sampler.sample(64, num_steps=num_steps, cfg_scale=2.0, class_labels=class_labels)
        
        samples = (samples + 1) / 2  # Scale to [0, 1]
        class_name = CIFAR10_CLASSES[class_idx]
        torchvision.utils.save_image(
            samples, output_dir / f'{model_type}_class_{class_idx}_{class_name}.png',
            nrow=8, normalize=False
        )
    
    print(f"✓ Saved sample grids to {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative models with FID and KID')
    parser.add_argument('--cfm_checkpoint', type=str, default=None,
                        help='Path to CFM checkpoint (optional if only evaluating DDPM)')
    parser.add_argument('--ddpm_checkpoint', type=str, default=None,
                        help='Path to DDPM checkpoint (optional if only evaluating CFM)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--num_fid_samples', type=int, default=10000,
                        help='Number of samples for FID calculation')
    parser.add_argument('--num_kid_samples', type=int, default=10000,
                        help='Number of samples for KID calculation')
    parser.add_argument('--kid_subsets', type=int, default=100,
                        help='Number of subsets for KID calculation')
    parser.add_argument('--kid_subset_size', type=int, default=1000,
                        help='Size of each subset for KID')
    parser.add_argument('--fid_steps', type=int, nargs='+', default=[50])
    parser.add_argument('--evaluate_fid', action='store_true', default=True,
                        help='Evaluate FID score')
    parser.add_argument('--evaluate_kid', action='store_true', default=True,
                        help='Evaluate KID score')
    parser.add_argument('--per_class_samples', type=int, default=1000,
                        help='Number of samples per class for per-class evaluation')
    parser.add_argument('--per_class_kid_subsets', type=int, default=100,
                        help='Number of subsets for per-class KID')
    parser.add_argument('--per_class_kid_subset_size', type=int, default=500,
                        help='Subset size for per-class KID (smaller than overall)')
    parser.add_argument('--balanced_classes', action='store_true', default=True,
                        help='Generate equal samples per class (1000 per class for 10k samples)')
    parser.add_argument('--random_classes', action='store_true', default=False,
                        help='Generate samples with random class distribution (overrides --balanced_classes)')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Validate that at least one checkpoint is provided
    if args.cfm_checkpoint is None and args.ddpm_checkpoint is None:
        parser.error("At least one of --cfm_checkpoint or --ddpm_checkpoint must be provided")
    
    # Handle balanced vs random classes
    balanced_classes = args.balanced_classes and not args.random_classes
    if balanced_classes:
        print(f"\n{'='*60}")
        print("Class Distribution: BALANCED")
        print(f"Will generate equal samples per class")
        print(f"  (e.g., for 10k samples: 1000 per class)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Class Distribution: RANDOM")
        print(f"Classes will be randomly sampled")
        print(f"{'='*60}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Determine which models to evaluate
    evaluate_cfm = args.cfm_checkpoint is not None
    evaluate_ddpm = args.ddpm_checkpoint is not None
    
    cfm_model = None
    ddpm_model = None
    cfm_sampler = None
    ddpm_sampler = None
    
    # Load models and create samplers
    print("=" * 60)
    if evaluate_cfm:
        print("Loading CFM model...")
        cfm_model, cfm_checkpoint = load_model(args.cfm_checkpoint, device)
        print("Creating CFM sampler...")
        cfm_sampler = CFMSampler(cfm_model, device)
    else:
        print("Skipping CFM model (no checkpoint provided)")
    
    if evaluate_ddpm:
        print("\nLoading DDPM model...")
        ddpm_model, ddpm_checkpoint = load_model(args.ddpm_checkpoint, device)
        print("Creating DDPM sampler...")
        ddpm_sampler = DDPMSampler(ddpm_model, device, num_timesteps=200)
    else:
        print("\nSkipping DDPM model (no checkpoint provided)")
    print("=" * 60)
    
    # Inference speed comparison
    if cfm_sampler:
        cfm_speed = measure_inference_speed(cfm_sampler, 'cfm')
        results['speed'] = {'cfm': cfm_speed}
        torch.cuda.empty_cache()
    
    if ddpm_sampler:
        ddpm_speed = measure_inference_speed(ddpm_sampler, 'ddpm')
        if 'speed' in results:
            results['speed']['ddpm'] = ddpm_speed
        else:
            results['speed'] = {'ddpm': ddpm_speed}
        torch.cuda.empty_cache()
    
    # Memory usage
    if cfm_sampler:
        cfm_memory = measure_memory_usage(cfm_sampler, 'cfm', device=device)
        results['memory'] = {'cfm_gb': cfm_memory}
        torch.cuda.empty_cache()
    
    if ddpm_sampler:
        ddpm_memory = measure_memory_usage(ddpm_sampler, 'ddpm', device=device)
        if 'memory' in results:
            results['memory']['ddpm_gb'] = ddpm_memory
        else:
            results['memory'] = {'ddpm_gb': ddpm_memory}
        torch.cuda.empty_cache()
    
    # Overall FID scores
    if args.evaluate_fid:
        results['fid_overall'] = {}
        for num_steps in args.fid_steps:
            step_results = {}
            
            if cfm_sampler:
                cfm_fid = evaluate_fid(cfm_sampler, 'cfm', args.num_fid_samples, 
                                       num_steps=num_steps, device=device,
                                       balanced_classes=balanced_classes)
                step_results['cfm'] = cfm_fid
                torch.cuda.empty_cache()
            
            if ddpm_sampler:
                ddpm_fid = evaluate_fid(ddpm_sampler, 'ddpm', args.num_fid_samples,
                                        num_steps=num_steps, device=device,
                                        balanced_classes=balanced_classes)
                step_results['ddpm'] = ddpm_fid
                torch.cuda.empty_cache()
            
            results['fid_overall'][f'steps_{num_steps}'] = step_results
    
    # Per-class FID scores
    if args.evaluate_fid:
        results['fid_per_class'] = {}
        for num_steps in args.fid_steps:
            step_results = {}
            
            if cfm_sampler:
                cfm_fid_per_class = evaluate_fid_per_class(
                    cfm_sampler, 'cfm', 
                    samples_per_class=args.per_class_samples,
                    num_steps=num_steps, 
                    device=device
                )
                step_results['cfm'] = cfm_fid_per_class
                torch.cuda.empty_cache()
            
            if ddpm_sampler:
                ddpm_fid_per_class = evaluate_fid_per_class(
                    ddpm_sampler, 'ddpm',
                    samples_per_class=args.per_class_samples,
                    num_steps=num_steps,
                    device=device
                )
                step_results['ddpm'] = ddpm_fid_per_class
                torch.cuda.empty_cache()
            
            results['fid_per_class'][f'steps_{num_steps}'] = step_results
    
    # Overall KID scores
    if args.evaluate_kid:
        results['kid_overall'] = {}
        for num_steps in args.fid_steps:
            step_results = {}
            
            if cfm_sampler:
                cfm_kid_mean, cfm_kid_std = evaluate_kid(
                    cfm_sampler, 'cfm', args.num_kid_samples,
                    num_steps=num_steps, 
                    num_subsets=args.kid_subsets,
                    subset_size=args.kid_subset_size,
                    device=device,
                    balanced_classes=balanced_classes
                )
                step_results['cfm'] = {
                    'mean': float(cfm_kid_mean) if cfm_kid_mean is not None else None,
                    'std': float(cfm_kid_std) if cfm_kid_std is not None else None,
                }
                torch.cuda.empty_cache()
            
            if ddpm_sampler:
                ddpm_kid_mean, ddpm_kid_std = evaluate_kid(
                    ddpm_sampler, 'ddpm', args.num_kid_samples,
                    num_steps=num_steps,
                    num_subsets=args.kid_subsets,
                    subset_size=args.kid_subset_size,
                    device=device,
                    balanced_classes=balanced_classes
                )
                step_results['ddpm'] = {
                    'mean': float(ddpm_kid_mean) if ddpm_kid_mean is not None else None,
                    'std': float(ddpm_kid_std) if ddpm_kid_std is not None else None,
                }
                torch.cuda.empty_cache()
            
            results['kid_overall'][f'steps_{num_steps}'] = step_results
    
    # Per-class KID scores
    if args.evaluate_kid:
        results['kid_per_class'] = {}
        for num_steps in args.fid_steps:
            step_results = {}
            
            if cfm_sampler:
                cfm_kid_per_class = evaluate_kid_per_class(
                    cfm_sampler, 'cfm',
                    samples_per_class=args.per_class_samples,
                    num_steps=num_steps,
                    num_subsets=args.per_class_kid_subsets,
                    subset_size=args.per_class_kid_subset_size,
                    device=device
                )
                step_results['cfm'] = cfm_kid_per_class
                torch.cuda.empty_cache()
            
            if ddpm_sampler:
                ddpm_kid_per_class = evaluate_kid_per_class(
                    ddpm_sampler, 'ddpm',
                    samples_per_class=args.per_class_samples,
                    num_steps=num_steps,
                    num_subsets=args.per_class_kid_subsets,
                    subset_size=args.per_class_kid_subset_size,
                    device=device
                )
                step_results['ddpm'] = ddpm_kid_per_class
                torch.cuda.empty_cache()
            
            results['kid_per_class'][f'steps_{num_steps}'] = step_results
    
    # Generate sample grids
    if cfm_sampler:
        generate_sample_grids(cfm_sampler, 'cfm', output_dir / 'cfm_samples', 
                             num_steps=args.fid_steps[0], device=device)
        torch.cuda.empty_cache()
    
    if ddpm_sampler:
        generate_sample_grids(ddpm_sampler, 'ddpm', output_dir / 'ddpm_samples',
                             num_steps=args.fid_steps[0], device=device)
        torch.cuda.empty_cache()
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    print(f"\n Results saved to: {results_file}")
    print(f" Sample grids saved to: {output_dir}")
    print("\nEvaluation complete!\n")


if __name__ == '__main__':
    main()