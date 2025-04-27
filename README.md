# UAE License Plate Recognition System

A sophisticated computer vision application for detecting and recognizing UAE license plates using YOLOv9.

## Project Overview

This project implements a license plate recognition system specifically designed for UAE license plates. It uses YOLOv9 for object detection and offers two detection approaches:

1. **Single-stage detection**: Directly detects and recognizes characters on license plates in a single pass
2. **Two-stage detection**: First detects the license plate in the image, then performs a second detection pass on the extracted plate region for more accurate character recognition

The system can identify the emirate, category, and plate number from UAE license plates.

## Features

- License plate detection and character recognition
- Two distinct detection methodologies (single-stage and two-stage)
- Batch processing for analyzing multiple images
- Visualization tools for viewing and comparing detection results
- Command-line interface for different processing modes
- Comprehensive Jupyter notebook for interactive experimentation

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- YOLOv9

## Project Structure

```
UAE_license_plates_detection/
├── outputs/              # Output visualization images
├── src/
│   ├── data/             # Dataset files and configuration
│   ├── detect/           # Detection implementations
│   │   ├── batch.py      # Batch processing module
│   │   ├── comparison.py # Comparison module
│   │   ├── single.py     # Single-stage detection module
│   │   └── two_stage.py  # Two-stage detection module
│   ├── models/           # YOLOv9 model definitions
│   ├── trained_models/   # Pre-trained model weights
│   ├── utils/            # Utility functions
│   │   └── visualization.py  # Visualization tools
│   ├── main.py           # Main CLI application entry point
│   └── license_plate_detection.ipynb  # Jupyter notebook implementation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/zg0ul/UAE_license_plates_detection
cd dubai_license
```

2. Install requirements:

```bash
pip install -r src/requirements.txt
```

3. Download pre-trained weights: [link](https://drive.google.com/drive/folders/1qYkh60rNdolhdqy899_xdKbdpJC22ztR?usp=sharing) and place them in `src/`

4. Download the plates_dataset: [link](https://drive.google.com/drive/folders/1-1aY2lN8HKkVNaPhnE1bEP4cZuqHkQKM?usp=sharing) and place them in `src/data`

## Usage

### Command Line Interface

The system can be used through a command-line interface with various modes:

```bash
# Single-stage detection on one image
python src/main.py --mode single --img path/to/image.jpg

# Two-stage detection on one image
python src/main.py --mode two-stage --img path/to/image.jpg

# Batch processing with single-stage detection
python src/main.py --mode batch --img-dir path/to/images/ --batch-size 5

# Batch processing with two-stage detection
python src/main.py --mode batch-two-stage --img-dir path/to/images/ --batch-size 5

# Compare single-stage and two-stage methods
python src/main.py --mode compare --img path/to/image.jpg
```

### Command Line Arguments

- `--weights`: Path to model weights (default: src/trained_models/gelan-c2/weights/best.pt)
- `--img-dir`: Directory containing test images (default: src/data/plates_dataset/test/images)
- `--img`: Path to a single image for processing
- `--batch-size`: Number of images to process in batch mode (default: 3)
- `--mode`: Processing mode [single, batch, two-stage, batch-two-stage, compare]
- `--conf-thres`: Confidence threshold for detections (default: 0.25)
- `--iou-thres`: IoU threshold for non-maximum suppression (default: 0.45)

### Jupyter Notebook

For interactive exploration and visualization, use the provided Jupyter notebook:

```bash
jupyter notebook src/license_plate_detection.ipynb
```

## Detection Approaches

### Single-Stage Detection

The single-stage approach detects license plate characters directly on the full image in a single pass. This method is faster but may be less accurate for images where the license plate is small or not clearly visible.

### Two-Stage Detection

The two-stage approach consists of:

1. First detection pass: Locate the license plate in the image
2. Plate extraction: Crop and process the detected license plate region
3. Second detection pass: Perform character recognition on the extracted plate image

This method typically provides better accuracy, especially for challenging images with smaller license plates.

## Output

The system outputs:

- Visualizations of detection results
- Extracted license plate information:
  - Emirate
  - Category
  - License plate number
- Detection confidence scores
- Annotated images showing bounding boxes around detected objects

## Acknowledgements

- [YOLOv9 for object detection](https://github.com/WongKinYiu/yolov9)
- The dataset is based on [Car-Plate collection](https://universe.roboflow.com/zara-hara/car-plate-k6xij) from Roboflow Universe

## Contact

<zg0ul.contact@gmail.com>
