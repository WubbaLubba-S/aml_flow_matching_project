"""
Comparison Demo: Original vs Fine-Tuned Model for Inpainting
Shows side-by-side results to visualize fine-tuning improvements

Usage:
    python demo_comparison.py \
        --checkpoint_original checkpoint_best_cfm.pt \
        --checkpoint_finetuned outputs/cfm_finetuned/checkpoint_finetuned_best.pt
"""

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent folder (The root of your project)
parent_dir = os.path.dirname(current_dir)

#Add paths so Python can see everything
sys.path.append(current_dir) 
sys.path.append(parent_dir)  



from models.unet import TinyUNet
from trainer.cfm_trainer import ConditionalFlowMatcher
from cfm_inpainting import (
    sample_cfm_inpainting,
    sample_cfm_inpainting_rk4,
    InpaintingMaskGenerator,
)


def load_model(checkpoint_path, device='cuda', model_name='Model'):
    """Load trained CFM model from checkpoint"""
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model args from checkpoint
    args = checkpoint.get('args', {})
    
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
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ {model_name} loaded (epoch {checkpoint.get('epoch', '?')})")
    if 'finetuned_for_inpainting' in checkpoint:
        print(f"  → Fine-tuned for inpainting!")
    
    return model


def get_curated_test_images(num_good=2, num_bad=2):
    """
    Get curated test images: some that typically work well and some that are harder
    
    Returns:
        good_images, good_labels: Images that typically inpaint well
        bad_images, bad_labels: Images that are harder to inpaint (more complex)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Classes that typically inpaint well (simple, structured)
    easy_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
    
    # Classes that are harder (organic, detailed)
    hard_classes = [2, 3, 5, 6]  # bird, cat, dog, frog
    
    # Find images
    good_images = []
    good_labels = []
    bad_images = []
    bad_labels = []
    
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        
        if label in easy_classes and len(good_images) < num_good:
            good_images.append(img)
            good_labels.append(label)
        
        if label in hard_classes and len(bad_images) < num_bad:
            bad_images.append(img)
            bad_labels.append(label)
        
        if len(good_images) >= num_good and len(bad_images) >= num_bad:
            break
    
    good_images = torch.stack(good_images)
    good_labels = torch.tensor(good_labels)
    bad_images = torch.stack(bad_images)
    bad_labels = torch.tensor(bad_labels)
    
    return good_images, good_labels, bad_images, bad_labels


def create_comparison_visualization(original, masked, mask, inpainted_orig, inpainted_ft, 
                                     labels, class_names, mask_name):
    """
    Create a comparison visualization showing:
    [Original | Masked | Mask | Original Model | Fine-tuned Model]
    """
    num_images = original.shape[0]
    
    # Create comparison grid
    comparison_rows = []
    
    for i in range(num_images):
        row = [
            original[i],        # Original image
            masked[i],          # Masked image
            mask[i].repeat(3, 1, 1),  # Mask (replicate to 3 channels for visualization)
            inpainted_orig[i],  # Original model inpainting
            inpainted_ft[i],    # Fine-tuned model inpainting
        ]
        
        comparison_rows.append(torch.stack(row, dim=0))
    
    # Stack all rows
    comparison_grid = torch.cat(comparison_rows, dim=0)
    
    # Create grid with labels
    grid = torchvision.utils.make_grid(
        comparison_grid, 
        nrow=5,  # 5 columns: Original, Masked, Mask, Original Model, Fine-tuned Model
        padding=4,
        pad_value=1.0
    )
    
    return grid


def run_comparison(args):
    """Run comparison between original and fine-tuned models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load both models
    print("="*60)
    model_original = load_model(args.checkpoint_original, device, "Original Model")
    model_finetuned = load_model(args.checkpoint_finetuned, device, "Fine-tuned Model")
    print("="*60 + "\n")
    
    # Initialize flow matcher
    flow_matcher = ConditionalFlowMatcher()
    
    # Get curated test images
    print(f"Loading curated test images...")
    good_images, good_labels, bad_images, bad_labels = get_curated_test_images(
        num_good=args.num_good, num_bad=args.num_bad
    )
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"✓ Loaded {len(good_images)} 'good' images (easy to inpaint):")
    print(f"  Classes: {[class_names[l.item()] for l in good_labels]}")
    print(f"✓ Loaded {len(bad_images)} 'bad' images (harder to inpaint):")
    print(f"  Classes: {[class_names[l.item()] for l in bad_labels]}")
    print()
    
    # Combine all images
    all_images = torch.cat([good_images, bad_images], dim=0).to(device)
    all_labels = torch.cat([good_labels, bad_labels], dim=0).to(device)
    
    # Mask generator
    mask_gen = InpaintingMaskGenerator()
    
    # Test each mask type
    mask_types = {
        'center': lambda shape: mask_gen.center_mask(shape, mask_size=16),
        'random_bbox': lambda shape: mask_gen.random_bbox_mask(shape, min_size=10, max_size=18),
        'irregular': lambda shape: mask_gen.random_irregular_mask(shape, num_strokes=5),
        'half': lambda shape: mask_gen.half_image_mask(shape, direction='left'),
    }
    
    print("="*60)
    print("RUNNING COMPARISONS")
    print("="*60)
    
    for mask_name, mask_fn in mask_types.items():
        print(f"\n{'='*60}")
        print(f"Processing: {mask_name.upper()} mask")
        print('='*60)
        
        # Generate mask
        masks = mask_fn(all_images.shape).to(device)
        masked_images = masks * all_images
        
        # Run inpainting with ORIGINAL model
        print(f"  Running original model...")
        if args.use_rk4:
            inpainted_orig = sample_cfm_inpainting_rk4(
                model=model_original,
                flow_matcher=flow_matcher,
                image=all_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=all_labels,
                num_classes=10,
            )
        else:
            inpainted_orig = sample_cfm_inpainting(
                model=model_original,
                flow_matcher=flow_matcher,
                image=all_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=all_labels,
                num_classes=10,
                resample_strategy=args.resample_strategy,
            )
        
        # Run inpainting with FINE-TUNED model
        print(f"  Running fine-tuned model...")
        if args.use_rk4:
            inpainted_ft = sample_cfm_inpainting_rk4(
                model=model_finetuned,
                flow_matcher=flow_matcher,
                image=all_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=all_labels,
                num_classes=10,
            )
        else:
            inpainted_ft = sample_cfm_inpainting(
                model=model_finetuned,
                flow_matcher=flow_matcher,
                image=all_images,
                mask=masks,
                device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                class_labels=all_labels,
                num_classes=10,
                resample_strategy=args.resample_strategy,
            )
        
        # Create comparison visualization
        print(f"  Creating visualization...")
        comparison_grid = create_comparison_visualization(
            all_images, masked_images, masks, inpainted_orig, inpainted_ft,
            all_labels, class_names, mask_name
        )
        
        # Normalize to [0, 1] for saving
        comparison_grid = (comparison_grid + 1) / 2
        
        # Save comparison
        save_path = output_dir / f'comparison_{mask_name}.png'
        torchvision.utils.save_image(comparison_grid, save_path)
        print(f"  ✓ Saved to {save_path}")
        
        # Also create labeled version with text
        create_labeled_comparison(
            all_images, masked_images, masks, inpainted_orig, inpainted_ft,
            all_labels, class_names, mask_name, output_dir, 
            good_count=len(good_images)
        )
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE!")
    print('='*60)
    print(f"\nAll comparisons saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - comparison_center.png")
    print(f"  - comparison_random_bbox.png")
    print(f"  - comparison_irregular.png")
    print(f"  - comparison_half.png")
    print(f"  - comparison_center_labeled.png (with annotations)")
    print(f"  - comparison_random_bbox_labeled.png (with annotations)")
    print(f"  - comparison_irregular_labeled.png (with annotations)")
    print(f"  - comparison_half_labeled.png (with annotations)")
    print(f"\nLayout: [Original | Masked | Mask | Original Model | Fine-tuned Model]")
    print(f"\nFirst {args.num_good} rows: Easy cases (should work well)")
    print(f"Last {args.num_bad} rows: Hard cases (fine-tuned should improve more)")


def create_labeled_comparison(original, masked, mask, inpainted_orig, inpainted_ft,
                               labels, class_names, mask_name, output_dir, good_count):
    """Create a labeled version with PIL for better readability"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Convert to numpy and denormalize
        def to_pil(tensor):
            img = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
            img = img.clamp(0, 1)
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(img)
        
        num_images = original.shape[0]
        img_size = 32
        padding = 10
        label_height = 30
        
        # Column headers
        headers = ["Original", "Masked", "Mask", "Original Model", "Fine-tuned Model"]
        
        # Create canvas
        canvas_width = 5 * (img_size + padding) + padding
        canvas_height = num_images * (img_size + padding) + padding + label_height
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Try to use a better font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw headers
        for col, header in enumerate(headers):
            x = padding + col * (img_size + padding) + img_size // 2
            draw.text((x, 5), header, fill='black', font=font, anchor='mt')
        
        # Draw images
        for row in range(num_images):
            y_offset = label_height + padding + row * (img_size + padding)
            
            # Determine if this is a "good" or "bad" case
            case_type = "EASY" if row < good_count else "HARD"
            case_color = "green" if row < good_count else "red"
            
            # Draw case label
            draw.text((5, y_offset + img_size // 2), case_type, fill=case_color, 
                     font=font_small, anchor='lm')
            
            # Draw class name
            class_name = class_names[labels[row].item()]
            draw.text((5, y_offset + img_size // 2 + 15), class_name, fill='blue', 
                     font=font_small, anchor='lm')
            
            # Original
            img = to_pil(original[row])
            canvas.paste(img, (padding, y_offset))
            
            # Masked
            img = to_pil(masked[row])
            canvas.paste(img, (padding + (img_size + padding), y_offset))
            
            # Mask (convert to RGB)
            mask_img = to_pil(mask[row].repeat(3, 1, 1))
            canvas.paste(mask_img, (padding + 2 * (img_size + padding), y_offset))
            
            # Original model result
            img = to_pil(inpainted_orig[row])
            canvas.paste(img, (padding + 3 * (img_size + padding), y_offset))
            
            # Fine-tuned model result
            img = to_pil(inpainted_ft[row])
            canvas.paste(img, (padding + 4 * (img_size + padding), y_offset))
        
        # Save
        save_path = output_dir / f'comparison_{mask_name}_labeled.png'
        canvas.save(save_path)
        print(f"  ✓ Saved labeled version to {save_path}")
        
    except Exception as e:
        print(f"  ⚠ Could not create labeled version: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Original vs Fine-tuned Model for Inpainting'
    )
    
    # Model checkpoints
    parser.add_argument('--checkpoint_original', type=str, required=True,
                        help='Path to original model checkpoint')
    parser.add_argument('--checkpoint_finetuned', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    
    # Test images
    parser.add_argument('--num_good', type=int, default=2,
                        help='Number of "good" (easy) test cases')
    parser.add_argument('--num_bad', type=int, default=2,
                        help='Number of "bad" (hard) test cases')
    
    # Inpainting parameters
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--use_rk4', action='store_true',
                        help='Use RK4 integration (better quality)')
    parser.add_argument('--resample_strategy', type=str, default='replace',
                        choices=['replace', 'repaint'],
                        help='Resampling strategy')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/comparison',
                        help='Output directory for comparison images')
    
    args = parser.parse_args()
    
    print("="*60)
    print("INPAINTING MODEL COMPARISON")
    print("="*60)
    print(f"Original model:    {args.checkpoint_original}")
    print(f"Fine-tuned model:  {args.checkpoint_finetuned}")
    print(f"Easy test cases:   {args.num_good}")
    print(f"Hard test cases:   {args.num_bad}")
    print(f"Total test cases:  {args.num_good + args.num_bad}")
    print(f"Mask types:        4 (center, bbox, irregular, half)")
    print(f"CFG scale:         {args.cfg_scale}")
    print(f"Num steps:         {args.num_steps}")
    print(f"Integration:       {'RK4' if args.use_rk4 else 'Euler'}")
    print("="*60 + "\n")
    
    run_comparison(args)


if __name__ == '__main__':
    main()