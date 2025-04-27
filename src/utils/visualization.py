"""
Visualization utilities for license plate detection.
This module provides functions for displaying detection results.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def display_single_result(result, save_output=True, save_dir="outputs", block_display=True):
    """
    Display the result of single-stage detection

    Args:
        result: Dictionary with detection results
        save_output: Whether to save images to disk
        save_dir: Directory to save output images
        block_display: Whether to block execution until plot window is closed
    """
    if not result:
        print("No result to display")
        return

    # Create output directory if saving is enabled
    if save_output:
        os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    orig_img_rgb = cv2.cvtColor(result['original_img'], cv2.COLOR_BGR2RGB)
    ax1.imshow(orig_img_rgb)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Annotated image with detections
    result_img_rgb = cv2.cvtColor(result['result_img'], cv2.COLOR_BGR2RGB)
    ax2.imshow(result_img_rgb)
    ax2.set_title("Detected Objects")
    ax2.axis('off')

    plt.tight_layout()
    plt.suptitle("License Plate Detection", fontsize=16)

    # Save the figure if enabled
    if save_output:
        output_path = os.path.join(save_dir, "single_stage_result.png")
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")

        # Save individual images as well
        cv2.imwrite(os.path.join(save_dir, "original.png"),
                    result['original_img'])
        cv2.imwrite(os.path.join(save_dir, "detection.png"),
                    result['result_img'])

    # Display plot and block until closed if requested
    plt.show(block=block_display)

    # Print license information
    license_info = result['license_info']
    print("License Plate Information:")
    print(f"Emirate: {license_info['emirate']}")
    print(f"Category: {license_info['category']}")
    print(f"License Number: {license_info['plate_number']}")


def display_two_stage_results(result, save_output=True, save_dir="outputs", block_display=True):
    """
    Display the result of two-stage detection

    Args:
        result: Dictionary with detection results
        save_output: Whether to save images to disk
        save_dir: Directory to save output images
        block_display: Whether to block execution until plot window is closed
    """
    if not result:
        print("No result to display")
        return

    # Create output directory if saving is enabled
    if save_output:
        os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    axs[0, 0].imshow(cv2.cvtColor(result['original_img'], cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    # Annotated original image with plate detection
    axs[0, 1].imshow(cv2.cvtColor(result['result_img'], cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Stage 1: License Plate Detection")
    axs[0, 1].axis('off')

    # Extracted plate image
    if 'plate_img' in result and result['plate_img'] is not None:
        axs[1, 0].imshow(cv2.cvtColor(result['plate_img'], cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Stage 2: Extracted Plate")
    else:
        axs[1, 0].imshow(np.ones((100, 200, 3), dtype=np.uint8) * 200)
        axs[1, 0].set_title("Plate Extraction Failed")
    axs[1, 0].axis('off')

    # Annotated plate image
    if 'annotated_plate_img' in result and result['annotated_plate_img'] is not None:
        axs[1, 1].imshow(cv2.cvtColor(
            result['annotated_plate_img'], cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Stage 3: Text Detection on Plate")
    else:
        axs[1, 1].imshow(np.ones((100, 200, 3), dtype=np.uint8) * 200)
        axs[1, 1].set_title("Text Detection Failed")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.suptitle("Two-Stage License Plate Detection", fontsize=16)

    # Save the figure if enabled
    if save_output:
        output_path = os.path.join(save_dir, "two_stage_result.png")
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")

        # Save individual images as well
        cv2.imwrite(os.path.join(save_dir, "original.png"),
                    result['original_img'])
        cv2.imwrite(os.path.join(save_dir, "plate_detection.png"),
                    result['result_img'])
        if 'plate_img' in result and result['plate_img'] is not None:
            cv2.imwrite(os.path.join(save_dir, "cropped_plate.png"),
                        result['plate_img'])
        if 'annotated_plate_img' in result and result['annotated_plate_img'] is not None:
            cv2.imwrite(os.path.join(
                save_dir, "plate_text_detection.png"), result['annotated_plate_img'])

    # Display plot and block until closed if requested
    plt.show(block=block_display)

    # Print license information
    license_info = result['license_info']
    print("License Plate Information:")
    print(f"Emirate: {license_info['emirate']}")
    print(f"Category: {license_info['category']}")
    print(f"License Number: {license_info['plate_number']}")


def compare_methods_display(single_result, two_stage_result, image_name=None, save_output=True, save_dir="outputs", block_display=True):
    """
    Display comparison between single-stage and two-stage detection results

    Args:
        single_result: Dictionary with single-stage detection results
        two_stage_result: Dictionary with two-stage detection results
        image_name: Name of the image being processed (optional)
        save_output: Whether to save images to disk
        save_dir: Directory to save output images
        block_display: Whether to block execution until plot window is closed
    """
    if not single_result or not two_stage_result:
        print("Missing results for comparison")
        return

    # Create output directory if saving is enabled
    if save_output:
        os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Single-stage detection
    single_img_rgb = cv2.cvtColor(
        single_result['result_img'], cv2.COLOR_BGR2RGB)
    axs[0, 0].imshow(single_img_rgb)
    axs[0, 0].set_title("Single-Stage Detection")
    axs[0, 0].axis('off')

    # Two-stage detection (full image with plate)
    two_stage_img_rgb = cv2.cvtColor(
        two_stage_result['result_img'], cv2.COLOR_BGR2RGB)
    axs[0, 1].imshow(two_stage_img_rgb)
    axs[0, 1].set_title("Two-Stage Detection (Full Image)")
    axs[0, 1].axis('off')

    # Original image
    orig_img_rgb = cv2.cvtColor(
        single_result['original_img'], cv2.COLOR_BGR2RGB)
    axs[1, 0].imshow(orig_img_rgb)
    axs[1, 0].set_title("Original Image")
    axs[1, 0].axis('off')

    # Cropped plate with character detection
    if 'success' in two_stage_result and two_stage_result['success'] and \
       'annotated_plate_img' in two_stage_result and two_stage_result['annotated_plate_img'] is not None:
        plate_img_rgb = cv2.cvtColor(
            two_stage_result['annotated_plate_img'], cv2.COLOR_BGR2RGB)
        axs[1, 1].imshow(plate_img_rgb)
        axs[1, 1].set_title("Two-Stage: Plate Characters")
    else:
        axs[1, 1].imshow(np.ones((100, 200, 3), dtype=np.uint8) * 200)
        axs[1, 1].set_title("Plate Character Detection Failed")
    axs[1, 1].axis('off')

    title = "Comparison: Single-Stage vs Two-Stage Detection"
    if image_name:
        title += f" - {image_name}"

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)

    # Save the figure if enabled
    if save_output:
        output_path = os.path.join(save_dir, "comparison_result.png")
        plt.savefig(output_path)
        print(f"Comparison visualization saved to: {output_path}")

    # Display plot and block until closed if requested
    plt.show(block=block_display)

    # Print license information comparison
    license_info_single = single_result['license_info']
    license_info_two_stage = two_stage_result['license_info']

    print("License Plate Information Comparison:")
    print(f"{'Parameter':<10} {'Single-Stage':<15} {'Two-Stage':<15}")
    print("-" * 40)
    print(
        f"{'Emirate':<10} {license_info_single['emirate']:<15} {license_info_two_stage['emirate']:<15}")
    print(
        f"{'Category':<10} {license_info_single['category']:<15} {license_info_two_stage['category']:<15}")
    print(
        f"{'Number':<10} {license_info_single['plate_number']:<15} {license_info_two_stage['plate_number']:<15}")
