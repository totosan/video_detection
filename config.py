import os
import logging # Import logging
from collections import deque # Keep deque if used elsewhere, otherwise remove

# --- Static Configuration ---

# Static and template folders
STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'templates'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Model and Stream Configuration (Moved from runtime)
YOLO_MODEL_PATH = "yolo12n.engine" # Or yolo12n.pt if not using TensorRT
RTSP_STREAM_URL = "rtsp://admin:QxT638_!1@192.168.0.55:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

# Tracking data defaults (if needed as constants)
MAX_TRACK_POINTS = 30

# --- Logging Configuration ---
DEBUG = False # Set to True for verbose debug logging, False for info/warnings only
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Get logger for config module

logger.info("Static configuration loaded.")

# --- Removed Runtime State ---
# model instance (loaded in DetectionSystem)
# track_history (managed by DetectionSystem/ObjectDetector)
# tracked_objects_info (managed by DetectionSystem/ObjectDetector)
# frame_queue (managed by DetectionSystem)
# latest_frame (managed by DetectionSystem)
# frame_lock (managed by DetectionSystem)
# latest_detections (managed by DetectionSystem)
# detections_lock (managed by DetectionSystem)
# latest_annotated_frame (managed by DetectionSystem)
# annotated_frame_lock (managed by DetectionSystem)
# stop_event (managed by DetectionSystem)