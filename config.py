import os
import time
import threading
import queue
import logging # Import logging
from ultralytics import YOLO
from collections import deque

# Static and template folders
STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'templates'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Logging configuration
DEBUG = False # Set to True for verbose debug logging, False for info/warnings only
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Get logger for config module

# Load YOLO model
try:
    model = YOLO("yolo12n.engine")
    logger.info("Loaded YOLO model.") # Use logger
except Exception as e:
    logger.exception("Failed to load YOLO model. Exiting.") # Use logger.exception
    exit()

# Tracking data
track_history = {}  # {track_id: deque(points)}
max_track_points = 30
tracked_objects_info = {}  # {track_id: {'name': str, 'last_seen': float, ...}}

# Queue for frames to be processed by detector
frame_queue = queue.Queue(maxsize=5)

# Shared data for latest frame and detections
latest_frame = None
frame_lock = threading.Lock()
latest_detections = {"results": None, "frame_shape": None} # Store results and original frame shape
detections_lock = threading.Lock()

# Shared variable to store the latest frame with annotations
latest_annotated_frame = None
annotated_frame_lock = threading.Lock()

# Control flag to signal threads to stop
stop_event = threading.Event()