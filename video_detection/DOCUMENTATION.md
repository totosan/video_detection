# Video Detection System Documentation

This document provides comprehensive information about the video detection system, including how to set it up, available APIs, and integration options.

## Table of Contents

- [Video Detection System Documentation](#video-detection-system-documentation)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Hardware Support](#hardware-support)
    - [Installation](#installation)
    - [Docker Setup](#docker-setup)
    - [Environment Variables](#environment-variables)
  - [Running the System](#running-the-system)
    - [Direct Execution](#direct-execution)
    - [Docker Execution](#docker-execution)
  - [API Documentation](#api-documentation)
    - [Web Interface](#web-interface)
    - [REST API Endpoints](#rest-api-endpoints)
      - [Video Feeds](#video-feeds)
      - [Snapshot Endpoints](#snapshot-endpoints)
      - [Detection and Tracking](#detection-and-tracking)
      - [System Control](#system-control)
      - [Object Filtering](#object-filtering)
    - [API Usage Examples](#api-usage-examples)
      - [Detect Objects in an Image](#detect-objects-in-an-image)
      - [Get Currently Tracked Objects](#get-currently-tracked-objects)
      - [Change Video Source](#change-video-source)
    - [ZeroMQ Interface](#zeromq-interface)
    - [MCP Server Integration](#mcp-server-integration)
  - [Configuration Options](#configuration-options)
    - [Video Sources](#video-sources)
    - [Model Options](#model-options)
    - [Object Filtering](#object-filtering-1)
  - [Integration Examples](#integration-examples)
    - [Python Client Example](#python-client-example)
    - [MCP Client Example](#mcp-client-example)

## Setup

### Prerequisites

The system requires the following components:

- Python 3.11+
- OpenCV
- YOLO object detection models
- Flask for the web server
- ZeroMQ for messaging
- Additional dependencies in `requirements.txt`

For CUDA/GPU acceleration:
- CUDA 11 (for Jetson Nano or other NVIDIA hardware)
- TensorRT (optional, for optimized inference)

### Hardware Support

The system is designed to work with:
- Standard CPUs (slower inference)
- NVIDIA GPUs (faster inference with CUDA)
- Jetson Nano (optimized for edge AI)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd ai-video-solution/video_detection
```

2. **Install dependencies**

```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv pip install -r requirements.txt
```

3. **Install CUDA (if using GPU)**

For Ubuntu 18.04:
```bash
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
pip3 install future
pip3 install -U --user wheel mock pillow
pip3 install testresources
pip3 install setuptools==58.3.0
pip3 install Cython
```

Additional CUDA setup instructions are available in the README.md file.

### Docker Setup

A Dockerfile is provided for containerized deployment:

1. **Build the Docker image**

```bash
cd video_detection/Docker
./build.sh
```

2. **Run with Docker**

The `run-docker.sh` script in the root directory provides a convenient way to start the container:

```bash
./run-docker.sh rtsp://example.com/stream
```

### Environment Variables

- `VIDEO_URL`: Specifies the video source (RTSP URL or camera index)
- `FLASK_RUN_HOST`: Host to bind the Flask server (default: '0.0.0.0')
- `FLASK_RUN_PORT`: Port for the Flask server (default: 5000)

## Running the System

### Direct Execution

To run the system directly:

```bash
python app.py
```

The server will start on http://localhost:5000 by default.

### Docker Execution

Using the provided script:

```bash
./run-docker.sh <video-source>
```

Where `<video-source>` is an RTSP URL or camera index.

## API Documentation

### Web Interface

A web interface is available at http://localhost:5000/ providing:

- Live video feed with detection overlays
- Object tracking visualization
- Control panel for:
  - Toggling object tracking display
  - Setting object filters
  - Selecting video sources
  - Toggling backend annotation processing

### REST API Endpoints

#### Video Feeds

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | Raw video stream (MJPEG) |
| `/video_feed_annotated` | GET | Annotated video with detection overlays (MJPEG) |

#### Snapshot Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/snapshot` | GET | Single JPEG snapshot from the video feed |
| `/raw_snapshot` | GET | Raw snapshot from the current video source |
| `/backend_snapshot` | GET | Snapshot with backend annotations |

#### Detection and Tracking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tracked_objects` | GET | List of currently tracked objects |
| `/api/track_history` | GET | History of tracked objects with optional images |
| `/api/current_detections` | GET | Current object detections with images |
| `/api/current_detections_light` | GET | Lightweight version without images |
| `/api/detect` | POST | Upload and detect objects in an image |

#### System Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System running status |
| `/api/cams` | GET | List available cameras |
| `/api/selected_videosource` | POST | Change video source |
| `/api/toggle_tracking` | POST | Toggle tracking display |
| `/api/tracking_status` | GET | Get tracking display status |
| `/api/backend_annotation/toggle` | POST | Toggle backend annotations |
| `/api/backend_annotation/status` | GET | Get backend annotation status |

#### Object Filtering

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/set_object_filter` | POST | Set filter for specific object types |
| `/api/get_object_filter` | GET | Get current object filter |

### API Usage Examples

#### Detect Objects in an Image

```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/api/detect
```

Response:
```json
{
  "detections": [
    {
      "box": [10.5, 20.3, 110.6, 180.2],
      "label": "person",
      "confidence": 0.91
    },
    {
      "box": [210.5, 120.3, 310.6, 280.2],
      "label": "car",
      "confidence": 0.87
    }
  ]
}
```

#### Get Currently Tracked Objects

```bash
curl http://localhost:5000/api/tracked_objects
```

Response:
```json
[
  {
    "id": 1,
    "name": "person",
    "time_since_seen": 0.2,
    "detection_image": "base64_encoded_image_data"
  },
  {
    "id": 2,
    "name": "car",
    "time_since_seen": 1.5,
    "detection_image": "base64_encoded_image_data"
  }
]
```

#### Change Video Source

```bash
curl -X POST -H "Content-Type: application/json" -d '{"source_identifier":"rtsp://example.com/stream"}' http://localhost:5000/api/selected_videosource
```

Response:
```json
{
  "message": "Successfully changed video source to rtsp://example.com/stream",
  "new_source": "rtsp://example.com/stream"
}
```

### ZeroMQ Interface

The system provides a ZeroMQ interface for high-performance communication:

- **Protocol**: Request-Reply pattern
- **Endpoints**: 
  - Image PUB/SUB socket for receiving images
  - Results REP socket for sending detection results
- **Message Format**:
  - Send image bytes
  - Send "detect" signal
  - Receive JSON detection results

### MCP Server Integration

The system integrates with Model Context Protocol (MCP) server for AI-powered interactions:

1. **Setup MCP Server**:

```bash
cd ai-video-solution/mcp_server
uv sync  # Install dependencies
```

2. **Run MCP Server Directly** (development):

```bash
mcp dev mcp_server.py
```

3. **Run with MCPO** (recommended for integration):

```bash
uvx mcpo --host localhost --port 8081 -- uv run --with mcp mcp run mcp_server.py
```

4. **Access MCP Documentation**:
   - OpenAPI docs available at: http://localhost:8081/docs
   - API endpoint: http://localhost:8081/invoke/{tool_name}

## Configuration Options

### Video Sources

The system supports multiple video source types:

1. **Camera Device**:
   - Use a numeric index (e.g., `0` for the default webcam)
   - Configure via `VIDEO_URL` environment variable

2. **RTSP Stream**:
   - Use an RTSP URL (e.g., `rtsp://example.com/stream`)
   - Configure via `VIDEO_URL` environment variable

3. **API Control**:
   - Use the `/api/selected_videosource` endpoint to change sources at runtime

### Model Options

The system can use different YOLO models:

1. **Default Model**:
   - Uses "yolo12n.pt" (nano model)
   - Configure via `YOLO_MODEL_PATH` in config.py

2. **Hardware Acceleration**:
   - Automatically uses TensorRT (.engine) if available
   - Falls back to PyTorch model on systems without CUDA
   - Uses MPS on macOS for acceleration

3. **Object Tracking**:
   - Uses custom tracker configuration in "small_object_tracker.yaml"

### Object Filtering

You can filter which objects are displayed:

1. **Web Interface**:
   - Use the "Object Filter" input on the web interface
   - Enter comma-separated object names (e.g., "person,car")

2. **API**:
   - Use the `/api/set_object_filter` endpoint
   - Send a JSON array of object names to filter

## Integration Examples

### Python Client Example

```python
import requests
import cv2
import numpy as np
import base64
import json

# Get system status
response = requests.get("http://localhost:5000/api/status")
status = response.json()
print(f"System running: {status['running']}")

# Get current detections
response = requests.get("http://localhost:5000/api/current_detections_light")
detections = response.json()
for detection in detections:
    print(f"Detected {detection['label']} with confidence {detection['confidence']}")

# Submit an image for detection
image_path = "test_image.jpg"
with open(image_path, "rb") as img:
    response = requests.post(
        "http://localhost:5000/api/detect",
        files={"image": img}
    )
    results = response.json()
    print(f"Found {len(results['detections'])} objects in the image")
```

### MCP Client Example

```python
import requests
import json

# Call MCP server via MCPO
def call_mcp_tool(tool_name, params={}):
    response = requests.post(
        f"http://localhost:8081/invoke/{tool_name}",
        json=params
    )
    return response.json()

# Get system status
status = call_mcp_tool("get_system_status")
print(f"System status: {status}")

# Get current detections
detections = call_mcp_tool("get_current_detections")
print(f"Current detections: {detections}")

# Get a snapshot
snapshot = call_mcp_tool("get_snapshot")
# The snapshot is returned as base64 encoded image data
```

---

This documentation provides an overview of how to use the video detection system. For more detailed information, refer to the source code or contact the development team.
