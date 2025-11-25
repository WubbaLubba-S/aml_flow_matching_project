"""
MeanFlow Model Evaluation Script
- FID score calculation (overall and per-class)
- KID (Kernel Inception Distance) calculation (overall and per-class)
- Inference speed comparison
- Memory usage analysis
- Qualitative sample grids

Key Difference: MeanFlow uses 1-step sampling (no ODE integration!)
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

# Import MeanFlow model and sampling
import sys
sys.path.append('.')

current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

# 3. Add paths so Python can see everything
sys.path.append(current_dir) # Allows seeing 'models'
sys.path.append(parent_dir)  # Allows seeing 'training'


from models.unet_mean import TinyUNetMeanFlow, count_parameters
from trainer.mean_trainer import sample_meanflow_1step

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class MeanFlowSampler:
    """Efficient MeanFlow sampling - 1-step only!"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        print("✓ MeanFlow sampler initialized (1-step sampling)")
    
    def sample(self, num_samples, cfg_scale=2.0, class_labels=None):
        """
        Sample using MeanFlow's 1-step method
        
        Note: num_steps parameter is ignored for MeanFlow - always 1 step!
        """
        return sample_meanflow_1step(
            self.model, num_samples=num_samples, num_classes=10,
            device=self.device, cfg_scale=cfg_scale, 
            class_labels=class_labels
        )


def load_meanflow_model(checkpoint_path, device='cuda'):
    """Load trained MeanFlow model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if checkpoint has model_type
    if 'model_type' in checkpoint:
        print(f"  Checkpoint model_type: {checkpoint['model_type']}")
    
    # Get model configuration from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            config = args
        else:
            config = vars(args) if hasattr(args, '__dict__') else {}
    else:
        print("  Warning: No 'args' found in checkpoint, using default configuration")
        config = {}

    
    # Initialize MeanFlow model with notebook architecture
    model = TinyUNetMeanFlow(
        in_channels=3,
        out_channels=3,
        base_channels=config.get('base_channels', 28),
        channel_mults=config.get('channel_mults', [1, 2, 2, 2]),
        num_res_blocks=config.get('num_res_blocks', 2),
        attention_resolutions=config.get('attention_resolutions', [16]),
        num_classes=10,
        dropout=config.get('dropout', 0.1),
    )
    
    # ==============================================================================
    # FIX: State Dict Adapter (Map Old Checkpoint Keys to New Model Architecture)
    # ==============================================================================
    raw_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    for key, value in raw_state_dict.items():
        new_key = key
        
        # 1. Rename the main block containers
        if "down_blocks" in new_key:
            new_key = new_key.replace("down_blocks", "encoder_blocks")
        elif "up_blocks" in new_key:
            new_key = new_key.replace("up_blocks", "decoder_blocks")
        elif "down_samples" in new_key:
            new_key = new_key.replace("down_samples", "downsamplers")
        elif "up_samples" in new_key:
            new_key = new_key.replace("up_samples", "upsamplers")

        # 2. Fix Attention Layers 
        # The old model had attention INSIDE the blocks (e.g., down_blocks.1.1)
        # The new model has attention OUTSIDE the blocks (e.g., encoder_attns.1)
        
        # Fix Encoder Attention (Old: encoder_blocks.1.1 -> New: encoder_attns.1)
        if "encoder_blocks.1.1." in new_key and any(x in new_key for x in ["qkv", "proj", "norm"]):
            new_key = new_key.replace("encoder_blocks.1.1.", "encoder_attns.1.")
            
        # Fix Decoder Attention (Old: decoder_blocks.2.2 -> New: decoder_attns.2)
        if "decoder_blocks.2.2." in new_key and any(x in new_key for x in ["qkv", "proj", "norm"]):
            new_key = new_key.replace("decoder_blocks.2.2.", "decoder_attns.2.")

        new_state_dict[new_key] = value
    # ==============================================================================

    # Load the modified state dict
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print("!! Automatic mapping failed. The architecture might be too different.")
        print(e)
        raise e

    model.to(device)
    model.eval()
    
    # Print checkpoint info
    print(f"✓ MeanFlow model loaded successfully")
    print(f"  Model: TinyUNetMeanFlow (dual time variables)")
    print(f"  Parameters: {count_parameters(model):,}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.4f}")
    
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
    
    # Compute kernels
    K_XX = kernel_fn(X, X)
    K_YY = kernel_fn(Y, Y)
    K_XY = kernel_fn(X, Y)
    
    # MMD^2
    mmd_squared = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) + \
                  (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) - \
                  2 * K_XY.sum() / (n * m)
    
    return mmd_squared


def get_real_data_loader(batch_size=100):
    """Get CIFAR-10 test set dataloader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return test_loader


def generate_samples(sampler, num_samples, cfg_scale=2.0, balanced_classes=True, device='cuda'):
    """
    Generate samples using MeanFlow sampler
    
    Note: num_steps is not needed - MeanFlow always uses 1 step!
    """
    all_samples = []
    
    if balanced_classes:
        # Generate equal number of samples per class
        samples_per_class = num_samples // 10
        remaining = num_samples % 10
        
        for class_idx in tqdm(range(10), desc="Generating class-balanced samples"):
            n = samples_per_class + (1 if class_idx < remaining else 0)
            if n == 0:
                continue
            
            class_labels = torch.full((n,), class_idx, dtype=torch.long, device=device)
            
            # Generate with MeanFlow (1-step!)
            samples = sampler.sample(n, cfg_scale=cfg_scale, class_labels=class_labels)
            all_samples.append(samples)
            
            torch.cuda.empty_cache()
    else:
        # Generate with random classes
        batch_size = 50
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating samples"):
            n = min(batch_size, num_samples - i * batch_size)
            
            # Generate with MeanFlow (1-step!)
            samples = sampler.sample(n, cfg_scale=cfg_scale)
            all_samples.append(samples)
            
            torch.cuda.empty_cache()
    
    samples = torch.cat(all_samples, dim=0)[:num_samples]
    return samples


def evaluate_fid(sampler, num_samples=5000, cfg_scale=2.0, device='cuda', balanced_classes=True):
    """
    Evaluate FID score for MeanFlow model
    
    Note: num_steps parameter removed - MeanFlow is always 1-step!
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING FID (MeanFlow 1-step sampling)")
    print(f"Samples: {num_samples}, CFG scale: {cfg_scale}")
    print(f"{'='*60}")
    
    # Load Inception model
    print("Loading Inception model...")
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    
    # Get real data
    print("Loading real CIFAR-10 test set...")
    test_loader = get_real_data_loader(batch_size=100)
    
    real_images = []
    for images, _ in tqdm(test_loader, desc="Loading real data"):
        real_images.append(images)
    real_images = torch.cat(real_images, dim=0)
    
    print(f"Computing statistics for {len(real_images)} real images...")
    mu_real, sigma_real = compute_fid_statistics(real_images, inception, device)
    
    # Generate fake samples (1-step!)
    print(f"Generating {num_samples} samples with MeanFlow (1-step)...")
    fake_images = generate_samples(sampler, num_samples, cfg_scale=cfg_scale, 
                                   balanced_classes=balanced_classes, device=device)
    
    print(f"Computing statistics for {len(fake_images)} generated images...")
    mu_fake, sigma_fake = compute_fid_statistics(fake_images, inception, device)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    print(f"\n✓ FID Score (1-step): {fid:.2f}")
    
    return fid


def evaluate_fid_per_class(sampler, samples_per_class=500, cfg_scale=2.0, device='cuda'):
    """
    Evaluate per-class FID scores for MeanFlow
    
    Note: Always 1-step sampling!
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING PER-CLASS FID (MeanFlow 1-step)")
    print(f"Samples per class: {samples_per_class}, CFG scale: {cfg_scale}")
    print(f"{'='*60}")
    
    # Load Inception model
    print("Loading Inception model...")
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    
    # Get real data by class
    print("Loading real CIFAR-10 test set by class...")
    test_loader = get_real_data_loader(batch_size=100)
    
    real_images_by_class = {i: [] for i in range(10)}
    for images, labels in tqdm(test_loader, desc="Loading real data"):
        for i in range(10):
            mask = labels == i
            if mask.any():
                real_images_by_class[i].append(images[mask])
    
    for i in range(10):
        real_images_by_class[i] = torch.cat(real_images_by_class[i], dim=0)
    
    # Compute per-class FID
    fid_per_class = {}
    
    for class_idx in range(10):
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nEvaluating class {class_idx}: {class_name}")
        
        # Real statistics
        real_imgs = real_images_by_class[class_idx]
        mu_real, sigma_real = compute_fid_statistics(real_imgs, inception, device)
        
        # Generate fake samples for this class (1-step!)
        class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
        fake_imgs = sampler.sample(samples_per_class, cfg_scale=cfg_scale, class_labels=class_labels)
        
        # Fake statistics
        mu_fake, sigma_fake = compute_fid_statistics(fake_imgs, inception, device)
        
        # Calculate FID
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        fid_per_class[class_name] = float(fid)
        
        print(f"  FID (1-step): {fid:.2f}")
        
        torch.cuda.empty_cache()
    
    avg_fid = np.mean(list(fid_per_class.values()))
    print(f"\n✓ Average per-class FID (1-step): {avg_fid:.2f}")
    
    return fid_per_class


def evaluate_kid(sampler, num_samples=5000, num_subsets=100, subset_size=1000,
                cfg_scale=2.0, device='cuda', balanced_classes=True):
    """
    Evaluate KID score for MeanFlow
    
    Note: Always 1-step sampling!
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING KID (MeanFlow 1-step)")
    print(f"Samples: {num_samples}, Subsets: {num_subsets}, Subset size: {subset_size}")
    print(f"{'='*60}")
    
    # Load Inception model
    print("Loading Inception model...")
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    
    # Get real data
    print("Loading real CIFAR-10 test set...")
    test_loader = get_real_data_loader(batch_size=100)
    
    real_images = []
    for images, _ in tqdm(test_loader, desc="Loading real data"):
        real_images.append(images)
    real_images = torch.cat(real_images, dim=0)
    
    print("Computing features for real images...")
    features_real = compute_inception_features(real_images, inception, device)
    
    # Generate fake samples (1-step!)
    print(f"Generating {num_samples} samples with MeanFlow (1-step)...")
    fake_images = generate_samples(sampler, num_samples, cfg_scale=cfg_scale,
                                   balanced_classes=balanced_classes, device=device)
    
    print("Computing features for generated images...")
    features_fake = compute_inception_features(fake_images, inception, device)
    
    # Calculate KID with multiple subsets
    print(f"Computing KID over {num_subsets} subsets...")
    kernel_fn = lambda X, Y: polynomial_kernel(X, Y, degree=3, gamma=None, coef=1.0)
    
    kid_scores = []
    for _ in tqdm(range(num_subsets), desc="Computing KID subsets"):
        mmd = calculate_mmd(features_real, features_fake, kernel_fn, subset_size)
        kid_scores.append(mmd)
    
    kid_mean = np.mean(kid_scores)
    kid_std = np.std(kid_scores)
    
    print(f"\n✓ KID Score (1-step): {kid_mean:.6f} ± {kid_std:.6f}")
    
    return kid_mean, kid_std


def evaluate_kid_per_class(sampler, samples_per_class=500, num_subsets=50, subset_size=500,
                           cfg_scale=2.0, device='cuda'):
    """
    Evaluate per-class KID scores for MeanFlow
    
    Note: Always 1-step sampling!
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING PER-CLASS KID (MeanFlow 1-step)")
    print(f"Samples per class: {samples_per_class}, Subsets: {num_subsets}")
    print(f"{'='*60}")
    
    # Load Inception model
    print("Loading Inception model...")
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    
    # Get real data by class
    print("Loading real CIFAR-10 test set by class...")
    test_loader = get_real_data_loader(batch_size=100)
    
    real_images_by_class = {i: [] for i in range(10)}
    for images, labels in tqdm(test_loader, desc="Loading real data"):
        for i in range(10):
            mask = labels == i
            if mask.any():
                real_images_by_class[i].append(images[mask])
    
    for i in range(10):
        real_images_by_class[i] = torch.cat(real_images_by_class[i], dim=0)
    
    # Compute per-class KID
    kid_per_class = {}
    kernel_fn = lambda X, Y: polynomial_kernel(X, Y, degree=3, gamma=None, coef=1.0)
    
    for class_idx in range(10):
        class_name = CIFAR10_CLASSES[class_idx]
        print(f"\nEvaluating class {class_idx}: {class_name}")
        
        # Real features
        real_imgs = real_images_by_class[class_idx]
        features_real = compute_inception_features(real_imgs, inception, device)
        
        # Generate fake samples for this class (1-step!)
        class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
        fake_imgs = sampler.sample(samples_per_class, cfg_scale=cfg_scale, class_labels=class_labels)
        
        # Fake features
        features_fake = compute_inception_features(fake_imgs, inception, device)
        
        # Calculate KID
        kid_scores = []
        for _ in range(num_subsets):
            mmd = calculate_mmd(features_real, features_fake, kernel_fn, subset_size)
            kid_scores.append(mmd)
        
        kid_mean = np.mean(kid_scores)
        kid_std = np.std(kid_scores)
        
        kid_per_class[class_name] = {
            'mean': float(kid_mean),
            'std': float(kid_std)
        }
        
        print(f"  KID (1-step): {kid_mean:.6f} ± {kid_std:.6f}")
        
        torch.cuda.empty_cache()
    
    avg_kid = np.mean([v['mean'] for v in kid_per_class.values()])
    print(f"\n✓ Average per-class KID (1-step): {avg_kid:.6f}")
    
    return kid_per_class


def measure_inference_speed(sampler, num_runs=100):
    """
    Measure inference speed for MeanFlow
    
    Note: MeanFlow is always 1-step, so this is very fast!
    """
    print(f"\n{'='*60}")
    print(f"MEASURING INFERENCE SPEED (MeanFlow 1-step)")
    print(f"Runs: {num_runs}")
    print(f"{'='*60}")
    
    device = sampler.device
    batch_size = 64
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = sampler.sample(batch_size, cfg_scale=2.0)
    
    torch.cuda.synchronize()
    
    # Measure
    print("Measuring speed...")
    times = []
    for _ in tqdm(range(num_runs), desc="Speed test"):
        start = time.time()
        _ = sampler.sample(batch_size, cfg_scale=2.0)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / mean_time
    
    print(f"\n✓ Inference Speed (1-step):")
    print(f"  Mean time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")
    
    return {
        'mean_time_ms': float(mean_time * 1000),
        'std_time_ms': float(std_time * 1000),
        'throughput_imgs_per_sec': float(throughput)
    }


def measure_memory_usage(sampler, device='cuda'):
    """Measure peak memory usage for MeanFlow"""
    print(f"\n{'='*60}")
    print(f"MEASURING MEMORY USAGE (MeanFlow 1-step)")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Generate samples
    batch_size = 64
    _ = sampler.sample(batch_size, cfg_scale=2.0)
    
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    
    print(f"\n✓ Peak memory usage (1-step): {peak_memory:.2f} GB")
    
    return float(peak_memory)


def generate_sample_grids(sampler, output_dir, cfg_scale=2.0, device='cuda'):
    """Generate sample grids for qualitative evaluation"""
    print(f"\n{'='*60}")
    print(f"GENERATING SAMPLE GRIDS (MeanFlow 1-step)")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Random samples
    print("Generating random samples...")
    random_samples = sampler.sample(64, cfg_scale=cfg_scale)
    random_samples = (random_samples + 1) / 2
    torchvision.utils.save_image(
        random_samples, output_dir / 'random_samples_1step.png',
        nrow=8, normalize=False
    )
    
    # Class-conditional samples
    print("Generating class-conditional samples...")
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        class_labels = torch.full((8,), class_idx, dtype=torch.long, device=device)
        samples = sampler.sample(8, cfg_scale=cfg_scale, class_labels=class_labels)
        samples = (samples + 1) / 2
        torchvision.utils.save_image(
            samples, output_dir / f'class_{class_idx}_{class_name}_1step.png',
            nrow=8, normalize=False
        )
    
    print(f"✓ Sample grids saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MeanFlow Model')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to MeanFlow checkpoint')
    
    # Evaluation options
    parser.add_argument('--evaluate_fid', action='store_true', default=True,
                       help='Evaluate FID scores')
    parser.add_argument('--evaluate_kid', action='store_true', default=True,
                       help='Evaluate KID scores')
    parser.add_argument('--evaluate_speed', action='store_true', default=True,
                       help='Measure inference speed')
    parser.add_argument('--evaluate_memory', action='store_true', default=True,
                       help='Measure memory usage')
    parser.add_argument('--generate_samples', action='store_true', default=True,
                       help='Generate sample grids')
    
    # FID/KID parameters
    parser.add_argument('--num_fid_samples', type=int, default=5000,
                       help='Number of samples for overall FID')
    parser.add_argument('--num_kid_samples', type=int, default=5000,
                       help='Number of samples for overall KID')
    parser.add_argument('--per_class_samples', type=int, default=500,
                       help='Samples per class for per-class metrics')
    parser.add_argument('--kid_subsets', type=int, default=100,
                       help='Number of subsets for KID calculation')
    parser.add_argument('--kid_subset_size', type=int, default=1000,
                       help='Subset size for KID calculation')
    parser.add_argument('--per_class_kid_subsets', type=int, default=50,
                       help='Number of subsets for per-class KID')
    parser.add_argument('--per_class_kid_subset_size', type=int, default=500,
                       help='Subset size for per-class KID')
    
    # Sampling parameters
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--balanced_classes', action='store_true', default=True,
                       help='Generate balanced samples across classes')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/meanflow',
                       help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Load MeanFlow model
    print("=" * 60)
    print("Loading MeanFlow model...")
    model, checkpoint = load_meanflow_model(args.checkpoint, device)
    print("Creating MeanFlow sampler (1-step)...")
    sampler = MeanFlowSampler(model, device)
    print("=" * 60)
    
    # Inference speed
    if args.evaluate_speed:
        speed_results = measure_inference_speed(sampler)
        results['speed'] = speed_results
        torch.cuda.empty_cache()
    
    # Memory usage
    if args.evaluate_memory:
        memory_gb = measure_memory_usage(sampler, device=device)
        results['memory_gb'] = memory_gb
        torch.cuda.empty_cache()
    
    # Overall FID score
    if args.evaluate_fid:
        fid_score = evaluate_fid(
            sampler, args.num_fid_samples,
            cfg_scale=args.cfg_scale, device=device,
            balanced_classes=args.balanced_classes
        )
        results['fid_overall'] = float(fid_score)
        torch.cuda.empty_cache()
    
    # Per-class FID scores
    if args.evaluate_fid:
        fid_per_class = evaluate_fid_per_class(
            sampler, samples_per_class=args.per_class_samples,
            cfg_scale=args.cfg_scale, device=device
        )
        results['fid_per_class'] = fid_per_class
        torch.cuda.empty_cache()
    
    # Overall KID score
    if args.evaluate_kid:
        kid_mean, kid_std = evaluate_kid(
            sampler, args.num_kid_samples,
            num_subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            cfg_scale=args.cfg_scale, device=device,
            balanced_classes=args.balanced_classes
        )
        results['kid_overall'] = {
            'mean': float(kid_mean),
            'std': float(kid_std)
        }
        torch.cuda.empty_cache()
    
    # Per-class KID scores
    if args.evaluate_kid:
        kid_per_class = evaluate_kid_per_class(
            sampler, samples_per_class=args.per_class_samples,
            num_subsets=args.per_class_kid_subsets,
            subset_size=args.per_class_kid_subset_size,
            cfg_scale=args.cfg_scale, device=device
        )
        results['kid_per_class'] = kid_per_class
        torch.cuda.empty_cache()
    
    # Generate sample grids
    if args.generate_samples:
        generate_sample_grids(sampler, output_dir / 'samples',
                            cfg_scale=args.cfg_scale, device=device)
        torch.cuda.empty_cache()
    
    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MEANFLOW EVALUATION SUMMARY (1-step sampling)")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Samples saved to: {output_dir}")
    print("\n🎉 MeanFlow Evaluation complete! 🎉\n")
    print("Note: All sampling was 1-step (no ODE integration required!)")
    print("=" * 60)


if __name__ == '__main__':
    main()