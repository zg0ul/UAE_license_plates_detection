from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt

from detect.single import process_license_plate_image


def process_license_plates_batch(model, test_dir, num_images=5, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process multiple license plate images and display results
    """
    # Ensure test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist")
        return None

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

    print(f"Processing {len(selected_images)} images from {test_dir}")

    # Process each image
    results = []
    for img_path in selected_images:
        print(f"\nProcessing image: {img_path.name}")
        result = process_license_plate_image(
            str(img_path), model, conf_threshold, iou_threshold, save_output=False)

        if result is None:
            print(f"Failed to process image: {img_path.name}")
            continue

        results.append(result)

        # Force clear any previous output

        # Display original and processed images side by side
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
        plt.suptitle(f"License Plate: {img_path.name}", fontsize=16)
        plt.show()

        # Display extracted license info directly under the images
        license_info = result['license_info']
        print("License Plate Information:")
        print(f"Emirate: {license_info['emirate']}")
        print(f"Category: {license_info['category']}")
        print(f"License Number: {license_info['plate_number']}")

    return results