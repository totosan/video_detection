# AI Video Solution Architecture

This document provides a high-level overview of the AI video solution's architecture and key features.

## 1. High-Level Architecture

The AI video solution is designed as a distributed system composed of two main components: a **Video Detection Service** and a **Frontend Application**. The entire system is container-friendly and can be deployed using Docker. A placeholder for a future **MCP Server** exists.

```
+------------------------+      +---------------------------+
|                        |      |                           |
|   Frontend (C#)        +----->|   Video Detection (Python)|
|   (Console App)        |      |   (Flask API)             |
|                        |      |                           |
+------------------------+      +---------------------------+
                                   |
                                   |
                         +------------------------+
                         |                        |
                         |   MCP Server (Python)  |
                         |   (Future Feature)     |
                         |                        |
                         +------------------------+
```

### 1.1. Video Detection Service

The core of the solution is the **Video Detection Service**, a Python application responsible for processing video streams and performing real-time object detection and segmentation.

-   **Technology:** Python, Flask, YOLO
-   **Functionality:**
    -   Captures video from a specified source.
    -   Uses a YOLO (You Only Look Once) model to detect objects in each frame.
    -   Supports object segmentation to create pixel-level masks for detected objects.
    -   Provides a RESTful API (using Flask) to expose detection results, annotated frames, and system status.
-   **Deployment:** Can be run as a standalone Python application or as a Docker container. Dockerfiles are provided for different architectures, including Intel and NVIDIA Jetson.

### 1.2. Frontend Application

The **Frontend** is a C# console application that acts as an intelligent client to the Video Detection Service.

-   **Technology:** C#
-   **AI Integration:** The frontend leverages the **Semantic Kernel** to orchestrate calls to various AI models. It is designed to be extensible and can connect to:
    -   Local models via **Ollama**.
    -   Models from **Hugging Face**.
    -   Cloud-based models through **Azure AI Foundry**.
-   **Functionality:**
    -   Orchestrates complex interactions with the detection data using AI.
    -   Communicates with the Video Detection Service's API to fetch detection data.
    -   Can be used to build advanced workflows based on the detected objects.

### 1.3. MCP Server (Future Feature)

The **MCP Server** is a planned supporting Python service. Its role will be to provide a "Model Context Protocol" for more advanced interactions, such as dynamic model switching or other control-plane operations. **Note: This component is not actively developed and is reserved for future use.**

-   **Technology:** Python

## 2. Key Features

From a user's perspective, the solution offers the following features:

### 2.1. Real-Time Object Detection

The system can identify and locate objects in a live video feed. The default model is trained to detect a variety of common objects.

### 2.2. Object Segmentation

In addition to drawing bounding boxes, the system can generate a pixel-perfect mask for each detected object, which is useful for more detailed analysis.

### 2.3. Object Tracking

The solution can track detected objects across frames, assigning a unique ID to each object. This is useful for counting objects or analyzing their movement over time.

### 2.4. Web-Based User Interface

A simple web interface, served by the **Video Detection Service**, allows users to view the output of the video detection service in real-time from a web browser.

### 2.5. REST API for Detections

Developers can integrate the video detection capabilities into their own applications by using the provided REST API. The API allows fetching the latest detection results in a structured format (JSON).

### 2.6. Snapshot Generation

The system can be configured to save snapshots of the video feed with the detections overlaid. This is useful for logging and review.

### 2.7. Cross-Platform Deployment

With Docker support, the application can be deployed on various platforms, including standard x86 servers (Intel) and edge devices like the NVIDIA Jetson, which is optimized for AI workloads.

## 3. Configuration and Feature Toggles

Feature toggles and settings for the different components of the application are managed in specific configuration files and via API endpoints.

### 3.1. Video Detection Service Configuration

Most of the backend features, such as which model to use, video source, and tracking parameters, can be configured in the `video_detection/config.py` file.

#### Key Feature Toggles:

1. **Object Tracking (On/Off)**
   - **Location:** Can be toggled via API endpoint
   - **How to toggle:** Send POST request to `/api/toggle_tracking`
   - **Check status:** GET request to `/api/tracking_status`
   - **Description:** Enables or disables object tracking across frames. When enabled, each detected object receives a unique ID that persists across frames.

2. **Detection vs. Segmentation Model**
   - **Location:** `video_detection/config.py`
   - **Configuration:**
     ```python
     YOLO_MODEL_PATH = "yolo11n.pt"        # For detection only
     # YOLO_MODEL_PATH = "yolo11n-seg.pt"  # For segmentation
     ```
   - **Description:** Toggle between standard object detection (bounding boxes) and segmentation (pixel-level masks).

3. **Video Source**
   - **Location:** `video_detection/.env` file or environment variable
   - **Configuration:**
     ```
     VIDEO_URL=<your_video_source>
     ```
   - **Options:**
     - RTSP stream URL (e.g., `rtsp://camera_ip:port/stream`)
     - Local camera index (e.g., `0` for default webcam)
     - Video file path
   - **Description:** Specifies the video input source for the detection system.

4. **Debug Logging**
   - **Location:** `video_detection/config.py`
   - **Configuration:**
     ```python
     DEBUG = True  # Enable verbose debug logging
     DEBUG = False # Only info/warnings/errors
     ```
   - **Description:** Controls the verbosity of logging output.

5. **Detection Filtering**
   - **Location:** Available via API endpoints
   - **Endpoints:**
     - `/api/current_detections` - Filtered detections
     - `/api/current_detections_unfiltered` - All detections
   - **Description:** Filter detections by object class, confidence threshold, or other criteria through API parameters.

### 3.2. Frontend and AI Services Configuration

The frontend application's settings, including the endpoints for AI services like Ollama, Hugging Face, and Azure AI, are configured directly within the `Frontend/Program.cs` file. This is where you can switch between different AI providers or update connection details.
