"""
Dubai License Plate Recognition System
Main entry point for the license plate recognition application.
"""

import os
import sys
import argparse
import cv2
import torch
import matplotlib.pyplot as plt
import contextlib
from pathlib import Path

# Now import YOLOv9 modules and local modules
from models.common import DetectMultiBackend
from utils.plots import colors

from detect.single import process_license_plate_image
from detect.batch import process_license_plates_batch
from detect.two_stage import process_two_stage_detection, process_two_stage_batch
from detect.comparison import compare_methods, compare_methods_batch
from utils.visualization import display_single_result, display_two_stage_results

# Configure matplotlib to display plots inline
plt.rcParams['figure.figsize'] = [16, 8]  # Set default figure size


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UAE License Plate Detection')
    parser.add_argument('--weights', type=str, default='src/trained_models/gelan-c2/weights/best.pt',
                        help='Model weights path')
    parser.add_argument('--img-dir', type=str, default='src/data/plates_dataset/test/images',
                        help='Directory containing test images')
    parser.add_argument('--img', type=str, default=None,
                        help='Path to a single image')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Number of images to process in batch mode')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'batch', 'two-stage', 'batch-two-stage', 'compare'],
                        help='Processing mode')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load model
        print(f"Loading model from {args.weights}")
        model = DetectMultiBackend(args.weights)
        model.eval()
        
        # Process according to the selected mode
        if args.mode == 'single':
            if args.img is None:
                print("Error: In single mode, --img must be specified")
                return
            
            print(f"Processing single image: {args.img}")
            result = process_license_plate_image(
                args.img, model, args.conf_thres, args.iou_thres, save_output=False)
            
            if result:
                display_single_result(result)
            else:
                print("No results obtained for the image")
        
        elif args.mode == 'batch':
            print(f"Processing batch of images from {args.img_dir}")
            results = process_license_plates_batch(
                model, args.img_dir, args.batch_size, args.conf_thres, args.iou_thres)
            
            if not results:
                print("No results obtained from batch processing")
        
        elif args.mode == 'two-stage':
            if args.img is None:
                print("Error: In two-stage mode, --img must be specified")
                return
                
            print(f"Processing single image with two-stage detection: {args.img}")
            result = process_two_stage_detection(
                args.img, model, args.conf_thres, args.iou_thres)
            
            if result:
                display_two_stage_results(result)
            else:
                print("No results obtained for the image")
        
        elif args.mode == 'batch-two-stage':
            print(f"Processing batch of images with two-stage detection from {args.img_dir}")
            results = process_two_stage_batch(
                model, args.img_dir, args.batch_size, args.conf_thres, args.iou_thres)
            
            if not results:
                print("No results obtained from two-stage batch processing")
        
        elif args.mode == 'compare':
            if args.img is None:
                print(f"Comparing methods on multiple images from {args.img_dir}")
                results = compare_methods_batch(model, args.img_dir, args.batch_size)
            else:
                print(f"Comparing methods on single image: {args.img}")
                result = compare_methods(args.img, model)
            
        print("Processing complete")

    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()