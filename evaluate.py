"""
Comprehensive Evaluation Script
- FID score calculation
- Inference speed comparison
- Memory usage analysis
- Qualitative sample grids
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

# Import models
import sys
sys.path.append('.')
from models.unet import TinyUNet
from training.cfm_trainer import ConditionalFlowMatcher, sample_cfm, sample_cfm_rk4
from training.ddpm_trainer import DDPMNoiseSchedule, DDPM, sample_ddpm


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint['args']
    
    # Initialize model
    model = TinyUNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.get('base_channels', 32),
        channel_mults=args.get('channel_mults', [1, 2, 2]),
        num_res_blocks=args.get('num_res_blocks', 2),
        attention_resolutions=args.get('attention_resolutions', [16]),
        num_classes=10,
        dropout=args.get('dropout', 0.1),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    
    return model, checkpoint


def compute_fid_statistics(images, inception_model, device='cuda', batch_size=50):
    """
    Compute mean and covariance of Inception features
    
    Args:
        images: (N, 3, 32, 32) tensor in [-1, 1] range
        inception_model: InceptionV3 feature extractor
        device: device to use
        batch_size: batch size for processing
    
    Returns:
        mu: mean of features
        sigma: covariance of features
    """
    inception_model.eval()
    features_list = []
    
    # Resize images to 299x299 for InceptionV3
    resize = transforms.Resize((299, 299), antialias=True)
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing features"):
            batch = images[i * batch_size : (i + 1) * batch_size].to(device)
            
            # Resize and normalize for Inception
            batch_resized = resize(batch)
            
            # Get features
            features = inception_model(batch_resized)
            features_list.append(features.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    
    # Compute statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two Gaussian distributions
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
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


def get_inception_model(device='cuda'):
    """Load pre-trained InceptionV3 for FID calculation"""
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
        print("ERROR: scipy is required for FID calculation. Install with: pip install scipy")
        return None


def evaluate_fid(model, model_type, num_samples=10000, batch_size=100, 
                 num_steps=50, device='cuda'):
    """
    Evaluate FID score
    
    Args:
        model: trained model
        model_type: 'cfm' or 'ddpm'
        num_samples: number of samples to generate
        batch_size: batch size for generation
        num_steps: number of sampling steps
        device: device to use
    
    Returns:
        fid_score: FID between generated and real CIFAR-10
    """
    print(f"\nEvaluating FID for {model_type.upper()} (num_steps={num_steps})...")
    
    # Load Inception model
    inception = get_inception_model(device)
    if inception is None:
        return None
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    generated_images = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating samples"):
        batch_size_curr = min(batch_size, num_samples - i * batch_size)
        
        if model_type == 'cfm':
            flow_matcher = ConditionalFlowMatcher()
            samples = sample_cfm(
                model, flow_matcher, batch_size_curr, 10,
                device=device, num_steps=num_steps, cfg_scale=2.0
            )
        elif model_type == 'ddpm':
            noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
            noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
            noise_schedule.alphas_cumprod_prev = noise_schedule.alphas_cumprod_prev.to(device)
            noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
            noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
            noise_schedule.posterior_variance = noise_schedule.posterior_variance.to(device)
            
            samples = sample_ddpm(
                model, noise_schedule, batch_size_curr, 10,
                device=device, num_steps=num_steps, cfg_scale=2.0
            )
        
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
    real_images = torch.stack([dataset[i][0] for i in indices])
    
    # Compute statistics
    print("Computing statistics for generated images...")
    mu_gen, sigma_gen = compute_fid_statistics(generated_images, inception, device)
    
    print("Computing statistics for real images...")
    mu_real, sigma_real = compute_fid_statistics(real_images, inception, device)
    
    # Calculate FID
    import scipy.linalg
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    
    print(f"FID Score: {fid_score:.2f}")
    
    return fid_score


def measure_inference_speed(model, model_type, num_samples=100, num_steps_list=[20, 50, 100], 
                            device='cuda'):
    """
    Measure inference speed for different numbers of sampling steps
    
    Returns:
        results: dict with timing information
    """
    print(f"\nMeasuring inference speed for {model_type.upper()}...")
    
    results = {}
    
    for num_steps in num_steps_list:
        # Warm-up
        if model_type == 'cfm':
            flow_matcher = ConditionalFlowMatcher()
            _ = sample_cfm(model, flow_matcher, 4, 10, device=device, num_steps=num_steps)
        elif model_type == 'ddpm':
            noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
            noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
            noise_schedule.alphas_cumprod_prev = noise_schedule.alphas_cumprod_prev.to(device)
            noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
            noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
            noise_schedule.posterior_variance = noise_schedule.posterior_variance.to(device)
            _ = sample_ddpm(model, noise_schedule, 4, 10, device=device, num_steps=num_steps)
        
        torch.cuda.synchronize()
        
        # Actual timing
        start_time = time.time()
        
        if model_type == 'cfm':
            flow_matcher = ConditionalFlowMatcher()
            _ = sample_cfm(model, flow_matcher, num_samples, 10, 
                          device=device, num_steps=num_steps)
        elif model_type == 'ddpm':
            _ = sample_ddpm(model, noise_schedule, num_samples, 10, 
                           device=device, num_steps=num_steps)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_sample = total_time / num_samples
        
        results[f'steps_{num_steps}'] = {
            'total_time': total_time,
            'time_per_sample': time_per_sample,
            'samples_per_second': num_samples / total_time,
        }
        
        print(f"  Steps={num_steps}: {time_per_sample:.3f}s/sample, "
              f"{num_samples/total_time:.2f} samples/s")
    
    return results


def measure_memory_usage(model, model_type, batch_size=32, device='cuda'):
    """Measure peak GPU memory usage during sampling"""
    print(f"\nMeasuring memory usage for {model_type.upper()}...")
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    # Sample with typical settings
    if model_type == 'cfm':
        flow_matcher = ConditionalFlowMatcher()
        _ = sample_cfm(model, flow_matcher, batch_size, 10, 
                      device=device, num_steps=50)
    elif model_type == 'ddpm':
        noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
        noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
        noise_schedule.alphas_cumprod_prev = noise_schedule.alphas_cumprod_prev.to(device)
        noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
        noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
        noise_schedule.posterior_variance = noise_schedule.posterior_variance.to(device)
        _ = sample_ddpm(model, noise_schedule, batch_size, 10, 
                       device=device, num_steps=50)
    
    torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    print(f"  Peak memory usage: {peak_memory:.2f} GB")
    
    return peak_memory


def generate_sample_grids(model, model_type, output_dir, device='cuda'):
    """Generate qualitative sample grids"""
    print(f"\nGenerating sample grids for {model_type.upper()}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Class-conditional samples
    print("  Generating class-conditional samples...")
    for class_idx in range(10):
        class_labels = torch.full((64,), class_idx, dtype=torch.long)
        
        if model_type == 'cfm':
            flow_matcher = ConditionalFlowMatcher()
            samples = sample_cfm(
                model, flow_matcher, 64, 10, device=device, 
                num_steps=50, cfg_scale=2.0, class_labels=class_labels
            )
        elif model_type == 'ddpm':
            noise_schedule = DDPMNoiseSchedule(num_timesteps=200)
            noise_schedule.alphas_cumprod = noise_schedule.alphas_cumprod.to(device)
            noise_schedule.alphas_cumprod_prev = noise_schedule.alphas_cumprod_prev.to(device)
            noise_schedule.sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod.to(device)
            noise_schedule.sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod.to(device)
            noise_schedule.posterior_variance = noise_schedule.posterior_variance.to(device)
            samples = sample_ddpm(
                model, noise_schedule, 64, 10, device=device,
                num_steps=100, cfg_scale=2.0, class_labels=class_labels
            )
        
        samples = (samples + 1) / 2  # Scale to [0, 1]
        torchvision.utils.save_image(
            samples, output_dir / f'{model_type}_class_{class_idx}.png',
            nrow=8, normalize=False
        )
    
    print(f"  Saved sample grids to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative models')
    parser.add_argument('--cfm_checkpoint', type=str, required=True,
                        help='Path to CFM checkpoint')
    parser.add_argument('--ddpm_checkpoint', type=str, required=True,
                        help='Path to DDPM checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--num_fid_samples', type=int, default=10000)
    parser.add_argument('--fid_steps', type=int, nargs='+', default=[50])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Load models
    print("=" * 60)
    print("Loading CFM model...")
    cfm_model, cfm_checkpoint = load_model(args.cfm_checkpoint, device)
    
    print("\nLoading DDPM model...")
    ddpm_model, ddpm_checkpoint = load_model(args.ddpm_checkpoint, device)
    print("=" * 60)
    
    # Inference speed comparison
    cfm_speed = measure_inference_speed(cfm_model, 'cfm', device=device)
    ddpm_speed = measure_inference_speed(ddpm_model, 'ddpm', device=device)
    
    results['speed'] = {
        'cfm': cfm_speed,
        'ddpm': ddpm_speed,
    }
    
    # Memory usage
    cfm_memory = measure_memory_usage(cfm_model, 'cfm', device=device)
    ddpm_memory = measure_memory_usage(ddpm_model, 'ddpm', device=device)
    
    results['memory'] = {
        'cfm_gb': cfm_memory,
        'ddpm_gb': ddpm_memory,
    }
    
    # FID scores
    results['fid'] = {}
    for num_steps in args.fid_steps:
        cfm_fid = evaluate_fid(cfm_model, 'cfm', args.num_fid_samples, 
                               num_steps=num_steps, device=device)
        ddpm_fid = evaluate_fid(ddpm_model, 'ddpm', args.num_fid_samples,
                                num_steps=num_steps, device=device)
        
        results['fid'][f'steps_{num_steps}'] = {
            'cfm': cfm_fid,
            'ddpm': ddpm_fid,
        }
    
    # Generate sample grids
    generate_sample_grids(cfm_model, 'cfm', output_dir / 'cfm_samples', device)
    generate_sample_grids(ddpm_model, 'ddpm', output_dir / 'ddpm_samples', device)
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()