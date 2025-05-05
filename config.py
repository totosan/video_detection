import os
from dotenv import load_dotenv
import logging # Import logging
from collections import deque # Keep deque if used elsewhere, otherwise remove

# --- Static Configuration ---

# Static and template folders
STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'templates'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Model and Stream Configuration (Moved from runtime)
YOLO_MODEL_PATH = "yolo12n.pt" # Default to the PyTorch model
# Load environment variables from .env file
load_dotenv()

# Get VIDEO_URL from environment variables
VIDEO_URL = os.getenv("VIDEO_URL", "")

RTSP_STREAM_URL = VIDEO_URL

# Tracking data defaults (if needed as constants)
MAX_TRACK_POINTS = 30

# --- Logging Configuration ---
DEBUG = False # Set to True for verbose debug logging, False for info/warnings only
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__) # Get logger for config module

logger.info("Static configuration loaded.")
