"""
Comparison module for license plate detection methods.
This module provides functions to compare different detection approaches.
"""

from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

from detect.single import process_license_plate_image
from detect.two_stage import process_two_stage_detection
from utils.visualization import compare_methods_display


def compare_methods(image_path, model):
    """
    Compare single-stage vs two-stage detection on the same image
    
    Args:
        image_path: Path to the image to process
        model: Loaded YOLOv9 model
        
    Returns:
        Dictionary with results from both methods
    """
    print(f"Processing image: {Path(image_path).name}")

    # Run single-stage detection
    print("\n===== SINGLE STAGE DETECTION =====")
    single_result = process_license_plate_image(image_path, model)

    # Run two-stage detection
    print("\n===== TWO STAGE DETECTION =====")
    two_stage_result = process_two_stage_detection(image_path, model)
    
    # Display comparison
    compare_methods_display(
        single_result, 
        two_stage_result, 
        image_name=Path(image_path).name
    )

    # Return both results for comparison
    return {
        'single_stage': single_result,
        'two_stage': two_stage_result
    }


def compare_methods_batch(model, test_dir, num_images=3):
    """
    Compare methods on multiple images
    
    Args:
        model: Loaded YOLOv9 model
        test_dir: Directory containing test images
        num_images: Number of images to process
        
    Returns:
        List of comparison results
    """
    # Get test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(test_dir).glob(f'*{ext}')))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return None
    
    # Choose random images or all if fewer than requested
    if len(image_files) > num_images:
        selected_images = random.sample(image_files, num_images)
    else:
        selected_images = image_files
    
    results = []
    for img_path in selected_images:
        print(f"\n\n==== Processing image: {img_path.name} ====")
        result = compare_methods(str(img_path), model)
        results.append(result)
    
    return results