# YOLOv11n Segmentation Model Integration

This document describes how to use the YOLOv11n segmentation model for object segmentation in this video detection system.

## Overview

Instead of just detecting object bounding boxes, the YOLOv11n-seg model provides pixel-level segmentation masks for each detected object. This allows for more precise object delineation and visualization.

## Setup

1. Download the YOLOv11n-seg model:
   ```bash
   ./download_seg_model.sh
   ```

2. Ensure you have installed all the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify that `config.py` is correctly pointing to the segmentation model:
   ```python
   YOLO_MODEL_PATH = "yolov11n-seg.pt"
   ```

## Features

- **Segmentation Masks**: The system now detects and visualizes object shapes as filled semi-transparent overlays.
- **Object Tracking**: Maintains consistent tracking IDs while displaying segmentation masks.
- **Visual Enhancement**: Provides better visual understanding of object boundaries.

## Technical Details

The segmentation is implemented through:

1. **Detection**: The YOLOv11n-seg model produces both bounding boxes and segmentation masks.
2. **Visualization**: Masks are rendered as semi-transparent colored overlays with an outline.
3. **Integration**: Full compatibility with the existing tracking system.

## Performance Considerations

- The segmentation model may require more computational resources than the basic detection model.
- If you experience performance issues, consider:
  - Reducing the input resolution
  - Increasing the processing interval
  - Using TensorRT optimization if available on your platform

## Troubleshooting

- If segmentation masks are not appearing, ensure:
  - The YOLOv11n-seg model is correctly downloaded
  - The config points to the correct model file
  - The required dependencies are installed
  - The system has sufficient GPU resources (if applicable)
