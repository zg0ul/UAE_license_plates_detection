"""
Two-stage license plate detection module.
This module implements a two-stage approach:
1. First detect license plates in the full image
2. Extract and process the plate regions for more accurate character recognition
"""

import os
import cv2
import tempfile
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.extract_license_info import extract_license_info
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors


def process_two_stage_detection(image_path, model, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    """
    Two-stage license plate detection:
    1. First detect just the license plate in the full image
    2. Extract and crop around the license plate
    3. Run the model again on the cropped plate for detailed text recognition

    This version only performs cropping without perspective correction or enhancement.

    Args:
        image_path: Path to the input image
        model: Loaded YOLOv9 model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        img_size: Image size for inference

    Returns:
        Dictionary with detection results and visualizations
    """
    # Stage 1: Detect the license plate
    print("Stage 1: Detecting license plate...")
    dataset = LoadImages(image_path, img_size=img_size)

    # Load the original image
    for path, img, img_original, _, _ in dataset:
        original_img = img_original.copy()
        break

    # Run first stage detection (focus on finding the plate)
    plates = []
    img = torch.tensor(img).to(model.device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = non_max_suppression(model(img), conf_threshold, iou_threshold)

    # Process detections to find license plates
    annotator_original = Annotator(original_img.copy())

    if pred[0] is not None and len(pred[0]) > 0:
        detections = pred[0]
        detections[:, :4] = scale_boxes(
            img.shape[2:], detections[:, :4], original_img.shape).round()

        for *xyxy, conf, cls_id in detections:
            cls_id = int(cls_id)
            cls_name = model.names[cls_id] if cls_id < len(
                model.names) else f"Class {cls_id}"

            # Check if this is a license plate detection
            if 'plate' in str(cls_name).lower():
                x1, y1, x2, y2 = map(int, xyxy)
                plates.append({
                    'class': cls_name,
                    'confidence': float(conf),
                    'bbox': (x1, y1, x2, y2)
                })

                # Annotate the original image
                label = f"{cls_name} {float(conf):.2f}"
                annotator_original.box_label(xyxy, label, colors(cls_id, True))

    # If no license plates found, return early
    if not plates:
        print("No license plates detected in first stage.")
        return {
            'success': False,
            'original_img': original_img,
            'result_img': annotator_original.result(),
            'license_info': {'emirate': 'Unknown', 'category': 'Unknown', 'plate_number': 'Unknown'}
        }

    # Sort plates by confidence and take the highest one
    plates.sort(key=lambda x: x['confidence'], reverse=True)
    plate = plates[0]
    print(f"Found license plate with confidence {plate['confidence']:.2f}")

    # Stage 2: Extract the license plate (simple crop with margin)
    print("Stage 2: Extracting license plate...")

    # Extract plate region with some margin
    x1, y1, x2, y2 = plate['bbox']
    h, w = y2 - y1, x2 - x1

    # Add margins (10% on each side)
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)

    # Ensure margins don't go out of bounds
    x1_margin = max(0, x1 - margin_x)
    y1_margin = max(0, y1 - margin_y)
    x2_margin = min(original_img.shape[1], x2 + margin_x)
    y2_margin = min(original_img.shape[0], y2 + margin_y)

    # Crop the plate region (simple crop, no perspective correction)
    plate_img = original_img[y1_margin:y2_margin, x1_margin:x2_margin]

    # Stage 3: Run detection on the extracted plate
    print("Stage 3: Detecting text on extracted plate...")

    # Create a temporary file for the plate image
    plate_detections = []
    annotated_plate_img = None

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, plate_img)

    try:
        # Run detection on the plate image
        plate_dataset = LoadImages(temp_path, img_size=img_size)

        for _, plate_tensor, plate_img_original, _, _ in plate_dataset:
            plate_tensor = torch.tensor(plate_tensor).to(
                model.device).float() / 255.0
            if plate_tensor.ndimension() == 3:
                plate_tensor = plate_tensor.unsqueeze(0)

            # Run model on the plate image
            plate_pred = non_max_suppression(model(
                plate_tensor), conf_threshold * 0.7, iou_threshold)

            # Annotator for plate image
            annotator_plate = Annotator(plate_img_original.copy())

            if plate_pred[0] is not None and len(plate_pred[0]) > 0:
                plate_det = plate_pred[0]
                plate_det[:, :4] = scale_boxes(
                    plate_tensor.shape[2:], plate_det[:, :4], plate_img_original.shape).round()

                for *xyxy, conf, cls_id in plate_det:
                    cls_id = int(cls_id)
                    cls_name = model.names[cls_id] if cls_id < len(
                        model.names) else f"Class {cls_id}"
                    confidence = float(conf)
                    bbox = [int(x) for x in xyxy]

                    # Skip 'plate' class in the second pass - we only want the characters
                    if 'plate' in str(cls_name).lower():
                        continue

                    plate_detections.append({
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })

                    # Annotate the plate image
                    label = f"{cls_name} {confidence:.2f}"
                    annotator_plate.box_label(
                        xyxy, label, colors(cls_id, True))

            # Get the annotated plate image
            annotated_plate_img = annotator_plate.result()
            break  # Only process the first image

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Extract license info from the plate detections
    license_info = extract_license_info(plate_detections)

    # Prepare the final annotated original image with license info
    final_img = annotator_original.result()

    # Add license info text to the original image
    cv2.putText(final_img, f"Emirate: {license_info['emirate']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(final_img, f"Category: {license_info['category']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(final_img, f"Plate Number: {license_info['plate_number']}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Detection complete! Found: {license_info}")

    return {
        'success': True,
        'original_img': original_img,
        'result_img': final_img,
        'plate_img': plate_img,  # Cropped plate without annotations
        'annotated_plate_img': annotated_plate_img,  # Annotated plate
        'license_info': license_info,
        'plate_detections': plate_detections
    }


def process_two_stage_batch(model, test_dir, num_images=5, conf_threshold=0.25, iou_threshold=0.45, save_output=True, block_display=True):
    """
    Process multiple images using two-stage detection and display results

    Args:
        model: Loaded YOLOv9 model
        test_dir: Directory containing test images
        num_images: Number of random images to process
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_output: Whether to save visualization images to disk
        block_display: Whether to block execution until plot windows are closed

    Returns:
        List of results dictionaries
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(test_dir).glob(f'*{ext}')))

    if not image_files:
        print(f"No images found in {test_dir}")
        return None

    # Select random images or all if fewer than requested
    import random
    if len(image_files) > num_images:
        selected_images = random.sample(image_files, num_images)
    else:
        selected_images = image_files

    print(
        f"Processing {len(selected_images)} images using two-stage detection...")

    # Process each image
    results = []
    for img_path in selected_images:
        print(f"\nProcessing image: {img_path.name}")
        result = process_two_stage_detection(
            str(img_path), model, conf_threshold, iou_threshold)

        if result is None:
            print(f"Failed to process image: {img_path.name}")
            continue

        results.append(result)

        # Display results
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        # Original image
        orig_img_rgb = cv2.cvtColor(result['original_img'], cv2.COLOR_BGR2RGB)
        axs[0].imshow(orig_img_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Annotated original image
        result_img_rgb = cv2.cvtColor(result['result_img'], cv2.COLOR_BGR2RGB)
        axs[1].imshow(result_img_rgb)
        axs[1].set_title("Detected License Plate")
        axs[1].axis('off')

        # Cropped and annotated plate
        if result['success'] and 'annotated_plate_img' in result and result['annotated_plate_img'] is not None:
            plate_img_rgb = cv2.cvtColor(
                result['annotated_plate_img'], cv2.COLOR_BGR2RGB)
            axs[2].imshow(plate_img_rgb)
            axs[2].set_title("Cropped Plate with Detections")
        else:
            # If plate extraction failed, show a blank panel
            axs[2].imshow(np.ones((100, 200, 3), dtype=np.uint8) * 200)
            axs[2].set_title("Plate Extraction Failed")
        axs[2].axis('off')

        plt.tight_layout()
        plt.suptitle(f"License Plate: {img_path.name}", fontsize=16)
        plt.subplots_adjust(top=0.9)

        # Save the figure if enabled
        if save_output:
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join(
                "outputs", f"two_stage_result_{img_path.stem}.png")
            plt.savefig(output_path)
            print(f"Visualization saved to: {output_path}")

        # Display plot and block until closed if requested
        plt.show(block=block_display)

        # Display extracted license info
        license_info = result['license_info']
        print("License Plate Information:")
        print(f"Emirate: {license_info['emirate']}")
        print(f"Category: {license_info['category']}")
        print(f"License Number: {license_info['plate_number']}")

        # Display detection details for second stage
        if result['success'] and result['plate_detections']:
            print("\nPlate Detection Details:")
            print(f"{'Class':<15} {'Confidence':<10} {'Bounding Box (x1,y1,x2,y2)'}")
            print("-" * 60)
            for det in result['plate_detections']:
                cls = det['class']
                conf = det['confidence']
                bbox = det['bbox']
                print(
                    f"{cls:<15} {conf:.2f}       ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})")

    return results
