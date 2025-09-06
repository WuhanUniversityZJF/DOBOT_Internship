"""
RT-DETRv2 Inference Script with Visualization
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.misc import dist_utils
from src.core import YAMLConfig
from src.solver import TASKS


def visualize_detections(image, outputs, score_threshold=0.3, line_thickness=2, font_size=0.8):
    """
    Visualize detections on the image
    Args:
        image: numpy array (H,W,3) in BGR format
        outputs: model outputs dict with keys 'boxes', 'scores', 'labels'
        score_threshold: minimum score to show detection
        line_thickness: bounding box thickness
        font_size: label font size
    Returns:
        visualized image
    """
    # Convert to numpy if not already
    boxes = outputs['boxes'].cpu().numpy() if isinstance(outputs['boxes'], torch.Tensor) else outputs['boxes']
    scores = outputs['scores'].cpu().numpy() if isinstance(outputs['scores'], torch.Tensor) else outputs['scores']
    labels = outputs['labels'].cpu().numpy() if isinstance(outputs['labels'], torch.Tensor) else outputs['labels']

    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green color

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=line_thickness)

        # Add label and confidence
        label_text = f'{label}:{score:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)

        cv2.rectangle(image, (x1, y1 - text_height - 10),
                      (x1 + text_width, y1), color, -1)
        cv2.putText(image, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), 1)

    return image


def main(args) -> None:
    """Main inference function"""
    dist_utils.setup_distributed(args.print_rank, args.print_method)

    # Load config
    cfg = YAMLConfig(args.config)
    cfg.device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize solver (same as training)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # Load checkpoint - modified to use solver.model.load_state_dict directly
    checkpoint = torch.load(args.weights, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load model weights - modified to use solver.detector instead of solver.model
    if hasattr(solver, 'detector'):
        solver.detector.load_state_dict(state_dict)
        solver.detector.eval()
    elif hasattr(solver, 'model'):
        solver.model.load_state_dict(state_dict)
        solver.model.eval()
    else:
        raise AttributeError("Solver object has neither 'detector' nor 'model' attribute")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process single image or directory
    if args.image_file:
        image_paths = [args.image_file]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

    for img_path in tqdm(image_paths, desc='Processing images'):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Get image dimensions
            orig_h, orig_w = image.shape[:2]

            # Preprocess (convert to RGB and normalize)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)

            # Resize and pad image to match model input size
            if hasattr(cfg, 'input_size'):
                input_size = cfg.input_size
                h, w = input_size
                scale = min(h / orig_h, w / orig_w)
                new_h, new_w = int(orig_h * scale), int(orig_w * scale)

                # Resize image
                resized_image = cv2.resize(image_rgb, (new_w, new_h))

                # Pad image
                pad_h = h - new_h
                pad_w = w - new_w
                padded_image = np.pad(resized_image,
                                      ((0, pad_h), (0, pad_w), (0, 0)),
                                      mode='constant', constant_values=114)

                image_tensor = torch.from_numpy(padded_image.astype(np.float32) / 255.0)

            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(cfg.device)

            # Inference - modified to use solver.detector if available
            with torch.no_grad():
                if hasattr(solver, 'detector'):
                    outputs = solver.detector(image_tensor)
                else:
                    outputs = solver.model(image_tensor)

            # Get first batch results (assuming batch size=1)
            detections = {
                'boxes': outputs[0]['boxes'],
                'scores': outputs[0]['scores'],
                'labels': outputs[0]['labels']
            }

            # Rescale boxes to original image size if we resized the image
            if hasattr(cfg, 'input_size'):
                detections['boxes'] = detections['boxes'] / scale

            # Visualize
            vis_image = visualize_detections(
                image.copy(),
                detections,
                score_threshold=args.score_threshold,
                line_thickness=args.line_thickness,
                font_size=args.font_size
            )

            # Save result
            output_path = Path(args.output_dir) / Path(img_path).name
            cv2.imwrite(str(output_path), vis_image)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETRv2 Inference Script')

    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='path to trained model weights')

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_file', type=str,
                             help='path to single image file')
    input_group.add_argument('--image_dir', type=str,
                             help='path to directory containing images')

    # Config and output
    parser.add_argument('--config', type=str,
                        default='configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml',
                        help='model config file path')
    parser.add_argument('--output_dir', type=str, default='./infer_results',
                        help='directory to save visualization results')

    # Visualization parameters
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='score threshold for visualization')
    parser.add_argument('--line_thickness', type=int, default=2,
                        help='bounding box line thickness')
    parser.add_argument('--font_size', type=float, default=0.8,
                        help='font size for labels')

    # Device and environment
    parser.add_argument('--device', type=str,
                        help='device to use (cuda/cpu)')
    parser.add_argument('--print_method', type=str, default='builtin',
                        help='print method')
    parser.add_argument('--print_rank', type=int, default=0,
                        help='print rank id')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank id')

    args = parser.parse_args()

    # Set default device if not specified
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)