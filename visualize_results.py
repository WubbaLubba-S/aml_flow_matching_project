"""
Comprehensive Visualization Script for Flow Matching Paper
Generates all figures including generation metrics and inpainting results

Usage:
    python visualize_all_results.py \
        --cfm_eval path/to/cfm/evaluation_results.json \
        --ddpm_eval path/to/ddpm/evaluation_results.json \
        --mean_eval path/to/mean/evaluation_results.json \
        --inpaint_base path/to/base/evaluation_results_all_masks.json \
        --inpaint_finetuned path/to/finetuned/evaluation_results_all_masks.json \
        --output_dir ./figures
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MASK_TYPES = ['center', 'random_bbox', 'irregular', 'half']
MASK_LABELS = ['Center', 'Random BBox', 'Irregular', 'Half Image']


def load_json(path):
    """Load JSON file"""
    if path is None:
        return None
    with open(path, 'r') as f:
        return json.load(f)


# ==============================================================================
# GENERATION METRICS PLOTS
# ==============================================================================

def plot_overall_fid_kid(cfm_eval, mean_eval, ddpm_eval, output_dir):
    """Bar chart of overall FID and KID"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = []
    fid_scores = []
    kid_scores = []
    colors = []
    
    if cfm_eval:
        models.append('CFM')
        fid_scores.append(cfm_eval['fid_overall']['steps_50']['cfm'])
        kid_scores.append(cfm_eval['kid_overall']['steps_50']['cfm']['mean'] * 1000)
        colors.append('#2ecc71')
    
    if mean_eval:
        models.append('MeanFlow')
        fid_scores.append(mean_eval['fid_overall']['steps_50']['mean'])
        kid_scores.append(mean_eval['kid_overall']['steps_50']['mean']['mean'] * 1000)
        colors.append('#3498db')
    
    if ddpm_eval:
        models.append('DDPM')
        fid_scores.append(ddpm_eval['fid_overall']['steps_50']['ddpm'])
        kid_scores.append(ddpm_eval['kid_overall']['steps_50']['ddpm']['mean'] * 1000)
        colors.append('#e74c3c')
    
    # FID plot
    bars1 = axes[0].bar(models, fid_scores, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('FID Score (lower is better)', fontsize=12)
    axes[0].set_title('Overall FID Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, fid_scores):
        axes[0].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # KID plot
    bars2 = axes[1].bar(models, kid_scores, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('KID Score × 1000 (lower is better)', fontsize=12)
    axes[1].set_title('Overall KID Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, kid_scores):
        axes[1].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_overall_metrics.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'generation_overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved generation_overall_metrics.pdf/png")


def plot_per_class_fid(cfm_eval, mean_eval, output_dir):
    """Bar chart comparing per-class FID scores"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cfm_fid = []
    mean_fid = []
    
    for cls in CIFAR10_CLASSES:
        if cfm_eval:
            cfm_fid.append(cfm_eval['fid_per_class']['steps_50']['cfm'].get(cls, 0))
        if mean_eval:
            mean_fid.append(mean_eval['fid_per_class']['steps_50']['mean'].get(cls, 0))
    
    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.35
    
    if cfm_fid:
        bars1 = ax.bar(x - width/2, cfm_fid, width, label='CFM', color='#2ecc71', alpha=0.8)
    if mean_fid:
        bars2 = ax.bar(x + width/2, mean_fid, width, label='MeanFlow', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=14)
    ax.set_ylabel('FID Score (lower is better)', fontsize=14)
    ax.set_title('Per-Class FID Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_per_class_fid.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'generation_per_class_fid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved generation_per_class_fid.pdf/png")


# ==============================================================================
# INPAINTING METRICS PLOTS
# ==============================================================================

def plot_inpainting_comparison(base_results, finetuned_results, output_dir):
    """Compare base vs fine-tuned inpainting performance"""
    if base_results is None or finetuned_results is None:
        print(" Skipping inpainting comparison (missing data)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract metrics
    base_nmse = [base_results['results'][m]['overall']['nmse_mean'] for m in MASK_TYPES]
    base_psnr = [base_results['results'][m]['overall']['psnr_mean'] for m in MASK_TYPES]
    base_ssim = [base_results['results'][m]['overall']['ssim_mean'] for m in MASK_TYPES]
    
    ft_nmse = [finetuned_results['results'][m]['overall']['nmse_mean'] for m in MASK_TYPES]
    ft_psnr = [finetuned_results['results'][m]['overall']['psnr_mean'] for m in MASK_TYPES]
    ft_ssim = [finetuned_results['results'][m]['overall']['ssim_mean'] for m in MASK_TYPES]
    
    x = np.arange(len(MASK_TYPES))
    width = 0.35
    
    # NMSE (lower is better)
    axes[0].bar(x - width/2, base_nmse, width, label='Base CFM', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + width/2, ft_nmse, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
    axes[0].set_ylabel('NMSE (lower is better)', fontsize=12)
    axes[0].set_title('NMSE Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MASK_LABELS, rotation=30, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # PSNR (higher is better)
    axes[1].bar(x - width/2, base_psnr, width, label='Base CFM', color='#e74c3c', alpha=0.8)
    axes[1].bar(x + width/2, ft_psnr, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
    axes[1].set_ylabel('PSNR (dB, higher is better)', fontsize=12)
    axes[1].set_title('PSNR Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MASK_LABELS, rotation=30, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # SSIM (higher is better)
    axes[2].bar(x - width/2, base_ssim, width, label='Base CFM', color='#e74c3c', alpha=0.8)
    axes[2].bar(x + width/2, ft_ssim, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
    axes[2].set_ylabel('SSIM (higher is better)', fontsize=12)
    axes[2].set_title('SSIM Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(MASK_LABELS, rotation=30, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inpainting_base_vs_finetuned.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'inpainting_base_vs_finetuned.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved inpainting_base_vs_finetuned.pdf/png")


def plot_inpainting_improvement(base_results, finetuned_results, output_dir):
    """Plot percentage improvement from fine-tuning"""
    if base_results is None or finetuned_results is None:
        print(" Skipping improvement plot (missing data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvements
    nmse_imp = []
    psnr_imp = []
    ssim_imp = []
    
    for m in MASK_TYPES:
        base_nmse = base_results['results'][m]['overall']['nmse_mean']
        ft_nmse = finetuned_results['results'][m]['overall']['nmse_mean']
        nmse_imp.append((ft_nmse - base_nmse) / base_nmse * 100)  # Negative is better
        
        base_psnr = base_results['results'][m]['overall']['psnr_mean']
        ft_psnr = finetuned_results['results'][m]['overall']['psnr_mean']
        psnr_imp.append((ft_psnr - base_psnr) / base_psnr * 100)
        
        base_ssim = base_results['results'][m]['overall']['ssim_mean']
        ft_ssim = finetuned_results['results'][m]['overall']['ssim_mean']
        ssim_imp.append((ft_ssim - base_ssim) / base_ssim * 100)
    
    x = np.arange(len(MASK_TYPES))
    width = 0.25
    
    bars1 = ax.bar(x - width, nmse_imp, width, label='NMSE (↓ better)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, psnr_imp, width, label='PSNR (↑ better)', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, ssim_imp, width, label='SSIM (↑ better)', color='#3498db', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mask Type', fontsize=14)
    ax.set_ylabel('Improvement (%)', fontsize=14)
    ax.set_title('Fine-tuning Improvement by Mask Type', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MASK_LABELS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inpainting_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'inpainting_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved inpainting_improvement.pdf/png")


def plot_inpainting_per_class_heatmap(finetuned_results, output_dir, metric='psnr'):
    """Heatmap of per-class inpainting performance"""
    if finetuned_results is None:
        print(f" Skipping {metric} heatmap (missing data)")
        return
    
    # Build matrix
    data = np.zeros((len(CIFAR10_CLASSES), len(MASK_TYPES)))
    
    metric_key = f'{metric}_mean'
    
    for j, mask in enumerate(MASK_TYPES):
        for i, cls in enumerate(CIFAR10_CLASSES):
            if cls in finetuned_results['results'][mask]['per_class']:
                data[i, j] = finetuned_results['results'][mask]['per_class'][cls][metric_key]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = 'RdYlGn' if metric in ['psnr', 'ssim'] else 'RdYlGn_r'
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    unit = ' (dB)' if metric == 'psnr' else ''
    cbar.ax.set_ylabel(f'{metric.upper()}{unit}', rotation=-90, va='bottom', fontsize=12)
    
    # Add labels
    ax.set_xticks(np.arange(len(MASK_TYPES)))
    ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(MASK_LABELS, fontsize=11)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Add text annotations
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(MASK_TYPES)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(f'Per-Class {metric.upper()} (Fine-tuned Model)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'inpainting_perclass_{metric}_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'inpainting_perclass_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved inpainting_perclass_{metric}_heatmap.pdf/png")


def plot_mask_difficulty_radar(finetuned_results, output_dir):
    """Radar chart showing mask difficulty across metrics"""
    if finetuned_results is None:
        print(" Skipping radar chart (missing data)")
        return
    
    # Normalize metrics (0-1 scale, higher = better)
    nmse_vals = [finetuned_results['results'][m]['overall']['nmse_mean'] for m in MASK_TYPES]
    psnr_vals = [finetuned_results['results'][m]['overall']['psnr_mean'] for m in MASK_TYPES]
    ssim_vals = [finetuned_results['results'][m]['overall']['ssim_mean'] for m in MASK_TYPES]
    
    # Normalize
    nmse_norm = [1 - (v - min(nmse_vals)) / (max(nmse_vals) - min(nmse_vals) + 1e-8) for v in nmse_vals]
    psnr_norm = [(v - min(psnr_vals)) / (max(psnr_vals) - min(psnr_vals) + 1e-8) for v in psnr_vals]
    ssim_norm = [(v - min(ssim_vals)) / (max(ssim_vals) - min(ssim_vals) + 1e-8) for v in ssim_vals]
    
    # Radar chart
    categories = ['NMSE\n(inverted)', 'PSNR', 'SSIM']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, mask in enumerate(MASK_TYPES):
        values = [nmse_norm[i], psnr_norm[i], ssim_norm[i]]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=MASK_LABELS[i], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_title('Mask Type Performance Comparison\n(normalized, higher = better)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inpainting_mask_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'inpainting_mask_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved inpainting_mask_radar.pdf/png")


def create_summary_table(cfm_eval, mean_eval, ddpm_eval, base_inpaint, ft_inpaint, output_dir):
    """Create comprehensive summary text file"""
    summary = []
    summary.append("=" * 80)
    summary.append("COMPREHENSIVE EVALUATION SUMMARY")
    summary.append("=" * 80)
    
    # Generation metrics
    summary.append("\n" + "=" * 80)
    summary.append("GENERATION METRICS (50 sampling steps)")
    summary.append("=" * 80)
    summary.append(f"{'Method':<15} {'FID':>10} {'KID×1000':>12} {'Speed (img/s)':>15}")
    summary.append("-" * 55)
    
    if cfm_eval:
        fid = cfm_eval['fid_overall']['steps_50']['cfm']
        kid = cfm_eval['kid_overall']['steps_50']['cfm']['mean'] * 1000
        speed = cfm_eval['speed']['cfm']['steps_50']['samples_per_second']
        summary.append(f"{'CFM':<15} {fid:>10.2f} {kid:>12.2f} {speed:>15.2f}")
    
    if mean_eval:
        fid = mean_eval['fid_overall']['steps_50']['mean']
        kid = mean_eval['kid_overall']['steps_50']['mean']['mean'] * 1000
        speed = mean_eval['speed']['mean']['steps_50']['samples_per_second']
        summary.append(f"{'MeanFlow':<15} {fid:>10.2f} {kid:>12.2f} {speed:>15.2f}")
    
    if ddpm_eval:
        fid = ddpm_eval['fid_overall']['steps_50']['ddpm']
        kid = ddpm_eval['kid_overall']['steps_50']['ddpm']['mean'] * 1000
        speed = ddpm_eval['speed']['ddpm']['steps_50']['samples_per_second']
        summary.append(f"{'DDPM':<15} {fid:>10.2f} {kid:>12.2f} {speed:>15.2f}")
    
    # Inpainting metrics
    if base_inpaint and ft_inpaint:
        summary.append("\n" + "=" * 80)
        summary.append("INPAINTING METRICS (Base vs Fine-tuned)")
        summary.append("=" * 80)
        
        summary.append(f"\n{'Mask Type':<15} {'NMSE (Base -> FT)':<20} {'PSNR (Baset -> FT)':<20} {'SSIM (Base -> FT)':<20}")
        summary.append("-" * 80)
        
        for mask, label in zip(MASK_TYPES, MASK_LABELS):
            base = base_inpaint['results'][mask]['overall']
            ft = ft_inpaint['results'][mask]['overall']
            
            nmse_str = f"{base['nmse_mean']:.2f}->{ft['nmse_mean']:.2f}"
            psnr_str = f"{base['psnr_mean']:.1f}->{ft['psnr_mean']:.1f}"
            ssim_str = f"{base['ssim_mean']:.3f}->{ft['ssim_mean']:.3f}"
            
            summary.append(f"{label:<15} {nmse_str:<20} {psnr_str:<20} {ssim_str:<20}")
        
        # Improvement summary
        summary.append("\n" + "-" * 80)
        summary.append("IMPROVEMENT FROM FINE-TUNING:")
        summary.append("-" * 80)
        
        for mask, label in zip(MASK_TYPES, MASK_LABELS):
            base = base_inpaint['results'][mask]['overall']
            ft = ft_inpaint['results'][mask]['overall']
            
            nmse_imp = (ft['nmse_mean'] - base['nmse_mean']) / base['nmse_mean'] * 100
            psnr_imp = (ft['psnr_mean'] - base['psnr_mean']) / base['psnr_mean'] * 100
            ssim_imp = (ft['ssim_mean'] - base['ssim_mean']) / base['ssim_mean'] * 100
            
            summary.append(f"{label:<15} NMSE: {nmse_imp:+.1f}%  PSNR: {psnr_imp:+.1f}%  SSIM: {ssim_imp:+.1f}%")
    
    summary.append("\n" + "=" * 80)
    
    summary_text = "\n".join(summary)
    
    with open(output_dir / 'complete_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n Saved complete_summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Generate all visualization figures')
    
    # Generation results
    parser.add_argument('--cfm_eval', type=str, default=None)
    parser.add_argument('--ddpm_eval', type=str, default=None)
    parser.add_argument('--mean_eval', type=str, default=None)
    
    # Inpainting results
    parser.add_argument('--inpaint_base', type=str, default=None,
                       help='Base model inpainting results (evaluation_results_all_masks.json)')
    parser.add_argument('--inpaint_finetuned', type=str, default=None,
                       help='Fine-tuned model inpainting results')
    
    parser.add_argument('--output_dir', type=str, default='./figures')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)
    
    # Load data
    cfm_eval = load_json(args.cfm_eval)
    ddpm_eval = load_json(args.ddpm_eval)
    mean_eval = load_json(args.mean_eval)
    base_inpaint = load_json(args.inpaint_base)
    ft_inpaint = load_json(args.inpaint_finetuned)
    
    # Generation plots
    print("\n--- GENERATION METRICS ---")
    if any([cfm_eval, mean_eval, ddpm_eval]):
        plot_overall_fid_kid(cfm_eval, mean_eval, ddpm_eval, output_dir)
        plot_per_class_fid(cfm_eval, mean_eval, output_dir)
    
    # Inpainting plots
    print("\n--- INPAINTING METRICS ---")
    plot_inpainting_comparison(base_inpaint, ft_inpaint, output_dir)
    plot_inpainting_improvement(base_inpaint, ft_inpaint, output_dir)
    plot_inpainting_per_class_heatmap(ft_inpaint, output_dir, metric='psnr')
    plot_inpainting_per_class_heatmap(ft_inpaint, output_dir, metric='ssim')
    plot_mask_difficulty_radar(ft_inpaint, output_dir)
    
    # Summary
    print("\n--- SUMMARY ---")
    create_summary_table(cfm_eval, mean_eval, ddpm_eval, base_inpaint, ft_inpaint, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()