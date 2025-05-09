import queue
import threading
import time
import logging
import platform # Import platform
import os       # Import os
import torch    # Import torch to check CUDA
from collections import deque
from ultralytics import YOLO

# Assuming FrameGrabber and ObjectDetector will be refactored to accept dependencies
from frame_grabber import FrameGrabber
from object_detector import ObjectDetector
from config import MAX_TRACK_POINTS, YOLO_MODEL_PATH, RTSP_STREAM_URL # Import static config

# Define the path to the custom tracker config
CUSTOM_TRACKER_CONFIG = "small_object_tracker.yaml"

# Add imports for annotation
import cv2
import numpy as np # Corrected import
from annotation_worker import AnnotationWorker

logger = logging.getLogger(__name__)

class DetectionSystem:
    def __init__(self):
        logger.info("Initializing DetectionSystem...")
        # --- Configuration ---
        self.rtsp_stream_url_config = RTSP_STREAM_URL # Store the raw config value
        self.yolo_model_path_config = YOLO_MODEL_PATH
        self.max_track_points = MAX_TRACK_POINTS

        # --- Determine Source Type and Identifier ---
        self.source_identifier = None
        self.source_type = None # "rtsp" or "device"

        if self.rtsp_stream_url_config.isdigit():
            try:
                self.source_identifier = int(self.rtsp_stream_url_config)
                self.source_type = "device"
                logger.info(f"Configured video source is a device index: {self.source_identifier}")
            except ValueError: # Should not happen if isdigit() is true, but as a safeguard
                logger.error(f"Could not parse '{self.rtsp_stream_url_config}' as a device index. Defaulting to RTSP interpretation.")
                self.source_identifier = self.rtsp_stream_url_config
                self.source_type = "rtsp"
        elif not self.rtsp_stream_url_config: # Handle empty string case
            logger.error("VIDEO_URL is empty. DetectionSystem cannot start without a video source.")
            # Consider raising an error or setting a state that prevents start()
            raise ValueError("VIDEO_URL (RTSP_STREAM_URL) is not configured.")
        else:
            self.source_identifier = self.rtsp_stream_url_config
            self.source_type = "rtsp"
            logger.info(f"Configured video source is an RTSP URL: {self.source_identifier}")
        # ---------------------------------------------

        # --- Determine Model Path based on TensorRT/CUDA availability ---
        model_path_to_load = self.yolo_model_path_config # Default to .pt
        can_use_tensorrt = False
        system_os = platform.system()

        # Check if CUDA is available (primary requirement for TensorRT with YOLO)
        try:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                # Prioritize TensorRT (.engine) if CUDA is available (typically on Linux/Jetson)
                # You might add more specific checks here if needed (e.g., import tensorrt)
                can_use_tensorrt = True
                logger.info("CUDA is available. Checking for TensorRT engine.")
            else:
                logger.info("CUDA not available.")
        except Exception as e:
             logger.warning(f"Could not check CUDA availability: {e}. Assuming TensorRT is not usable.")
             cuda_available = False # Ensure it's False if check fails

        if can_use_tensorrt:
            # Try to find/use the .engine file
            base_name = os.path.splitext(self.yolo_model_path_config)[0]
            engine_path = base_name + ".engine"
            if os.path.exists(engine_path):
                model_path_to_load = engine_path
                logger.info(f"Using TensorRT engine: {model_path_to_load}")
            else:
                logger.warning(f"CUDA available, but TensorRT engine not found at {engine_path}. Falling back to PyTorch model: {self.yolo_model_path_config}")
                model_path_to_load = self.yolo_model_path_config # Explicitly set fallback
        else:
             # Log reason for using PyTorch model
             if system_os == "Darwin":
                 logger.info(f"macOS detected. Using PyTorch model: {model_path_to_load}")
             elif not cuda_available:
                 logger.info(f"CUDA not available on {system_os}. Using PyTorch model: {model_path_to_load}")
             else:
                 # Should not happen with current logic, but as a safeguard
                 logger.info(f"Using PyTorch model ({model_path_to_load}) due to other reasons (OS: {system_os}, CUDA: {cuda_available}).")
             # .pt model is already the default (model_path_to_load)

        # --- Model Loading ---
        try:
            self.model = YOLO(model_path_to_load)  # Load the determined YOLO model
            logger.info(f"Loaded YOLO model from {model_path_to_load}")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model from {model_path_to_load}. Cannot initialize DetectionSystem.")
            raise  # Re-raise the exception to prevent system from starting incorrectly

        # --- Runtime State ---
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_detections_data = {"results": None, "frame_shape": None, "track_history": {}, "tracked_objects_info": {}}
        self.backend_annotation_enabled = False # Flag for backend annotation

        # Dictionary to store detection images by track ID
        self.detection_images = {}
        self.detection_images_lock = threading.Lock()

        # --- Synchronization Primitives ---
        self.frame_lock = threading.Lock()
        self.annotated_frame_lock = threading.Lock()
        self.detections_data_lock = threading.Lock()
        self.backend_annotation_lock = threading.Lock() # Lock for the flag
        self.stop_event = threading.Event()

        # --- Queues ---
        self.frame_queue = queue.Queue(maxsize=2) # Queue for frames awaiting detection
        self.annotation_queue = queue.Queue(maxsize=2)  # Queue for frames awaiting annotation

        # --- Worker Threads ---
        self.frame_grabber = None
        self.object_detector = None
        self.annotation_worker = None
        logger.info("DetectionSystem initialized.")

    # --- State Update Callbacks/Methods ---
    # These methods will be passed to the worker threads to update the central state safely

    def update_latest_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame
            # logger.debug("DetectionSystem: Updated latest_frame") # Optional: can be noisy

    # Modified: Only updates raw detection data, no annotated frame here
    def update_detection_results(self, detections, frame_shape, track_history, tracked_objects_info, detection_images=None):
        with self.detections_data_lock:
            self.latest_detections_data["results"] = detections
            self.latest_detections_data["frame_shape"] = frame_shape
            self.latest_detections_data["track_history"] = track_history
            self.latest_detections_data["tracked_objects_info"] = tracked_objects_info

            # Add detailed debugging for detection_images
            if detection_images is None:
                if not hasattr(self, '_detection_images_warning_logged'):
                    logger.warning("detection_images parameter is None")
                    self._detection_images_warning_logged = True
            elif not isinstance(detection_images, dict):
                logger.warning(f"detection_images is not a dictionary: {type(detection_images)}")
            elif not detection_images:
                logger.warning("detection_images dictionary is empty")
            else:
                logger.debug(f"Received detection_images with {len(detection_images)} entries: {list(detection_images.keys())}")

            # Update detection images if provided
            if detection_images:
                with self.detection_images_lock:
                    stored_count = 0
                    error_count = 0
                    for track_id, image in detection_images.items():
                        if image is None:
                            logger.warning(f"Skipping None image for track_id {track_id}")
                            continue
                        try:
                            # Check if this is actually an image
                            if not isinstance(image, np.ndarray):
                                logger.warning(f"Image for track_id {track_id} is not a numpy array: {type(image)}")
                                error_count += 1
                                continue
                            if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                                logger.warning(f"Empty image for track_id {track_id}, shape: {image.shape}")
                                continue
                            # Store the valid image
                            self.detection_images[track_id] = image
                            stored_count += 1
                            logger.debug(f"✅ Stored valid image for track_id {track_id}, shape: {image.shape}")
                        except Exception as e:
                            logger.error(f"Error storing image for track_id {track_id}: {e}")
                            error_count += 1
                    logger.debug(f"Detection image storage summary: stored {stored_count}, errors {error_count}")

                # Optional: Clean up old images that are no longer in tracked_objects_info
                current_track_ids = set(tracked_objects_info.keys())
                old_track_ids = set(self.detection_images.keys()) - current_track_ids
                cleaned_count = 0
                for old_id in old_track_ids:
                    # Keep images for a while even after tracking is lost (optional)
                    last_seen = time.time() - 30  # Default to 30 seconds ago if not found
                    if old_id in tracked_objects_info:
                        last_seen = tracked_objects_info[old_id].get('last_seen', time.time())
                    # Remove if not seen for more than 30 seconds
                    if time.time() - last_seen > 30:
                        del self.detection_images[old_id]
                        cleaned_count += 1
                logger.debug(f"Cleaned up {cleaned_count} old images")
                logger.debug(f"DetectionSystem state: {len(self.detection_images)} images in storage")

    # New callback for the annotation worker
    def update_latest_annotated_frame(self, annotated_frame):
        with self.annotated_frame_lock:
            self.latest_annotated_frame = annotated_frame

    # --- Getters for Flask App ---
    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_annotated_frame(self):
        with self.annotated_frame_lock:
            return self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None

    def get_tracked_objects_info(self):
        with self.detections_data_lock:
            return self.latest_detections_data["tracked_objects_info"].copy() # Make a deep copy to be safe

    def get_track_history(self):
        with self.detections_data_lock:
            return self.latest_detections_data["track_history"].copy()

    # --- New Getter for Current Detections ---
    def get_current_detections_data(self):
        """Returns the latest raw detection results, frame shape, and track history."""
        with self.detections_data_lock:
            detections = self.latest_detections_data.get("results", [])
            shape = self.latest_detections_data.get("frame_shape")
            track_history_deques = self.latest_detections_data.get("track_history", {})
            tracked_objects_info = self.latest_detections_data.get("tracked_objects_info", {})

            frame_width = None
            frame_height = None
            if shape is not None and isinstance(shape, (tuple, list)) and len(shape) >= 2:
                frame_height = shape[0]
                frame_width = shape[1]

            # Convert deques to lists for JSON serialization
            track_history_lists = {
                str(track_id): list(points)
                for track_id, points in track_history_deques.items()
            }

            # Serialize YOLO Results as per UI expectation
            serializable_detections = []
            if detections and hasattr(detections, "__getitem__"):
                for det in detections:
                    if hasattr(det, "boxes") and det.boxes is not None:
                        boxes = det.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        track_ids = boxes.id.cpu().numpy() if hasattr(boxes, "id") and boxes.id is not None else [None]*len(xyxy)
                        for i in range(len(xyxy)):
                            box = [float(x) for x in xyxy[i]]
                            conf = float(confs[i])
                            cls_idx = int(clss[i])
                            track_id = int(track_ids[i]) if track_ids[i] is not None else None
                            label = self.model.names[cls_idx] if cls_idx < len(self.model.names) else "unknown"
                            # Color: try to get from tracked_objects_info, else fallback
                            color = tracked_objects_info.get(track_id, {}).get('color', [(track_id or 0)*50%255, (track_id or 0)*80%255, (track_id or 0)*120%255])
                            serializable_detections.append({
                                "box": box,
                                "conf": conf,
                                "cls": cls_idx,
                                "label": label,
                                "track_id": track_id,
                                "color": color
                            })
            return {
                "detections": serializable_detections,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "track_history": track_history_lists
            }
    # -----------------------------------------

    # --- Backend Annotation Control ---
    def enable_backend_annotation(self):
        with self.backend_annotation_lock:
            self.backend_annotation_enabled = True
            logger.info("Backend annotation ENABLED.")

    def disable_backend_annotation(self):
        with self.backend_annotation_lock:
            self.backend_annotation_enabled = False
            logger.info("Backend annotation DISABLED.")

    def toggle_backend_annotation(self):
        with self.backend_annotation_lock:
            self.backend_annotation_enabled = not self.backend_annotation_enabled
            status = "ENABLED" if self.backend_annotation_enabled else "DISABLED"
            logger.info(f"Backend annotation toggled: {status}.")
            return self.backend_annotation_enabled

    def is_backend_annotation_enabled(self):
        with self.backend_annotation_lock:
            return self.backend_annotation_enabled
    # ----------------------------------

    # --- Lifecycle Management ---
    def start(self):
        if self.frame_grabber or self.object_detector or self.annotation_worker:
            logger.warning("Detection system already running or not properly stopped.")
            return

        logger.info("Starting DetectionSystem threads...")
        self.stop_event.clear()

        # Clear queues and reset state asynchronously
        threading.Thread(target=self._clear_queues_and_reset_state, daemon=True).start()

        # Instantiate workers, passing necessary dependencies and callbacks
        self.frame_grabber = FrameGrabber(
            source_identifier=self.source_identifier,
            source_type=self.source_type,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            frame_update_callback=self.update_latest_frame # Pass the callback
        )
        self.object_detector = ObjectDetector(
            model=self.model,
            frame_queue=self.frame_queue,
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            results_update_callback=self.update_detection_results,
            max_track_points=self.max_track_points,
            model_names=self.model.names,
            tracker_config_path=CUSTOM_TRACKER_CONFIG # Pass the custom config path
        )
        self.annotation_worker = AnnotationWorker(
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            annotated_frame_callback=self.update_latest_annotated_frame,
            max_track_points=self.max_track_points,
            model_names=self.model.names,
            is_backend_annotation_enabled_func=self.is_backend_annotation_enabled # Pass the getter method
        )

        # Start threads asynchronously
        threading.Thread(target=self._start_workers, daemon=True).start()

    def _clear_queues_and_reset_state(self):
        # Clear queues and reset state
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        while not self.annotation_queue.empty():
            try: self.annotation_queue.get_nowait()
            except queue.Empty: break
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_detections_data = {"results": None, "frame_shape": None, "track_history": {}, "tracked_objects_info": {}}

    def _start_workers(self):
        logger.info("Starting FrameGrabber thread...")
        self.frame_grabber.start()
        time.sleep(0.1)
        logger.info("FrameGrabber thread started.")

        logger.info("Starting ObjectDetector thread...")
        self.object_detector.start()
        time.sleep(0.1)
        logger.info("ObjectDetector thread started.")

        logger.info("Starting AnnotationWorker thread...")
        self.annotation_worker.start()
        logger.info("AnnotationWorker thread started.")

        if self.frame_grabber.is_alive() and self.object_detector.is_alive() and self.annotation_worker.is_alive():
            logger.info("DetectionSystem threads started successfully.")
        else:
            logger.error("One or more DetectionSystem threads failed to start.")
            self.stop()

    def stop(self):
        logger.info("Stopping DetectionSystem threads...")
        self.stop_event.set()

        # Stop annotation worker
        if self.annotation_worker:
            logger.debug("Stopping annotation worker...") # Added logging
            self.annotation_worker.stop() # Call the worker's stop method
            self.annotation_worker = None

        # Stop detector first (depends on frame queue)
        if self.object_detector:
            logger.debug("Stopping object detector...")
            self.object_detector.stop() # Call the worker's stop method
            self.object_detector = None

        # Stop grabber
        if self.frame_grabber:
            logger.debug("Stopping frame grabber...")
            self.frame_grabber.stop() # Assuming FrameGrabber also has a stop method
            self.frame_grabber = None

        # Clear cached images
        with self.detection_images_lock:
            self.detection_images.clear()

        logger.info("DetectionSystem threads stopped.")

    def is_running(self):
        """Check if worker threads are alive."""
        grabber_alive = self.frame_grabber and self.frame_grabber.is_alive()
        detector_alive = self.object_detector and self.object_detector.is_alive()
        annotator_alive = self.annotation_worker and self.annotation_worker.is_alive()
        return grabber_alive and detector_alive and annotator_alive

    def get_detection_image(self, track_id):
        """Get the detection image for a specific track ID.
        
        Args:
            track_id: The ID of the tracked object.
        
        Returns:
            np.ndarray: The image of the detected object, or None if not available.
        """
        with self.detection_images_lock:
            image = self.detection_images.get(track_id)
            if image is not None:
                logger.debug(f"Retrieved detection image for track ID {track_id}, shape: {image.shape}")
            else:
                logger.debug(f"No detection image found for track ID {track_id}")
            return image

    def get_latest_detection_image(self):
        """Returns the latest detection image for use in the API."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

