import os
import time
import torch
import cv2
from pathlib import Path
from utils.plots import Annotator, colors
from utils.general import non_max_suppression, scale_boxes
from utils import extract_license_info
from utils.dataloaders import LoadImages


def process_license_plate_image(image_path, model, conf_threshold=0.25, iou_threshold=0.45, img_size=640, save_output=False):
    """
    Process a license plate image: detect objects, extract license info, and visualize results

    Args:
        image_path: Path to the license plate image
        model: Loaded YOLOv9 model
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        img_size: Image size for inference
        save_output: Whether to save the output image (default: False)

    Returns:
        Dictionary with extracted license info and visualization
    """
    # Load and process image
    start_time = time.time()

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist")
        return None

    try:
        dataset = LoadImages(image_path, img_size=img_size)

        all_detections = []
        license_info = None
        result_img = None
        original_img = None

        for path, img, img_original, _, _ in dataset:
            # Store original image
            original_img = img_original.copy()

            # Run inference
            img = torch.tensor(img).to(model.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Get predictions
            pred = non_max_suppression(
                model(img), conf_threshold, iou_threshold)

            # Process detections
            annotator = Annotator(img_original.copy())

            if pred[0] is not None and len(pred[0]) > 0:
                detections = pred[0]
                # Scale boxes to original image dimensions
                detections[:, :4] = scale_boxes(
                    img.shape[2:], detections[:, :4], img_original.shape).round()

                # Process each detection
                for *xyxy, conf, cls_id in detections:
                    cls_id = int(cls_id)
                    cls_name = model.names[cls_id] if cls_id < len(
                        model.names) else f"Class {cls_id}"
                    confidence = float(conf)
                    bbox = [int(x) for x in xyxy]

                    # Add detection to list
                    all_detections.append({
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })

                    # Draw box on image
                    label = f"{cls_name} {confidence:.2f}"
                    annotator.box_label(xyxy, label, colors(cls_id, True))

            # Extract license information
            license_info = extract_license_info(all_detections)

            # Get the annotated image
            result_img = annotator.result()

            # Save output if requested (now disabled by default)
            if save_output:
                output_dir = 'outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_name = Path(path).stem
                output_path = os.path.join(
                    output_dir, f"license_info_{file_name}.jpg")
                cv2.imwrite(output_path, result_img)
                print(f"Saved result to {output_path}")

            # Break after first image (in case of video input)
            break

        # Calculate processing time
        processing_time = time.time() - start_time

        return {
            'license_info': license_info,
            'original_img': original_img,
            'result_img': result_img,
            'detections': all_detections,
            'processing_time': processing_time
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None