# Using YOLO Segmentation with the Video Detection System

This guide explains how to use the YOLO segmentation model in the video detection system for more precise object detection.

## What is Object Segmentation?

Unlike traditional object detection that provides rectangular bounding boxes, segmentation provides pixel-precise masks that outline the exact shape of detected objects. This is particularly useful when:

- Objects have irregular shapes
- Objects partially occlude each other
- More precise object boundaries are required

## Getting Started

1. Download the YOLOv11n-seg model:
   ```bash
   cd video_detection
   ./download_seg_model.sh
   ```

2. The system should automatically use the segmentation model based on the configuration in `config.py`:
   ```python
   YOLO_MODEL_PATH = "yolov11n-seg.pt"
   ```

3. Restart your detection system to apply the changes.

## How It Works

The segmentation workflow follows these steps:

1. The YOLO model detects objects and generates segmentation masks
2. For each detected object, both a bounding box and a segmentation mask are stored
3. The annotation worker draws the segmentation masks as semi-transparent colored overlays
4. If segmentation isn't available for any reason, the system falls back to bounding boxes

## Performance Considerations

Segmentation requires more processing power than simple bounding box detection. If you experience performance issues:

- On resource-constrained devices, you might need to adjust processing parameters
- Enable TensorRT optimization if available on your platform
- Consider reducing the frame resolution or processing frequency

## Troubleshooting

If segmentation masks aren't showing:

1. Check that `yolov11n-seg.pt` exists in your working directory
2. Verify that the `config.py` points to the segmentation model
3. Make sure you have all required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

For more detailed information, refer to the full SEGMENTATION_README.md file.
