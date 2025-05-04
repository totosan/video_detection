import queue
import threading
import time
import logging
from collections import deque
from ultralytics import YOLO

# Assuming FrameGrabber and ObjectDetector will be refactored to accept dependencies
from frame_grabber import FrameGrabber
from object_detector import ObjectDetector
from config import MAX_TRACK_POINTS, YOLO_MODEL_PATH, RTSP_STREAM_URL # Import static config

# Add imports for annotation
import cv2
import numpy as np # Corrected import
from annotation_worker import AnnotationWorker

logger = logging.getLogger(__name__)

class DetectionSystem:
    def __init__(self):
        logger.info("Initializing DetectionSystem...")
        # --- Configuration ---
        self.rtsp_stream_url = RTSP_STREAM_URL
        self.yolo_model_path = YOLO_MODEL_PATH
        self.max_track_points = MAX_TRACK_POINTS

        # --- Model Loading ---
        try:
            self.model = YOLO(self.yolo_model_path)  # Load the YOLO model
            self.model.devices = 'mps'  # Set device to MPS (Metal Performance Shaders) for macOS
            logger.info(f"Loaded YOLO model from {self.yolo_model_path}")
        except Exception as e:
            logger.exception("Failed to load YOLO model. Cannot initialize DetectionSystem.")
            raise  # Re-raise the exception to prevent system from starting incorrectly

        # --- Runtime State ---
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_detections_data = {"results": None, "frame_shape": None, "track_history": {}, "tracked_objects_info": {}}

        # --- Synchronization Primitives ---
        self.frame_lock = threading.Lock()
        self.annotated_frame_lock = threading.Lock()
        self.detections_data_lock = threading.Lock()
        self.stop_event = threading.Event()

        # --- Queues ---
        self.frame_queue = queue.Queue(maxsize=5) # Queue for frames awaiting detection
        self.annotation_queue = queue.Queue(maxsize=10)  # Queue for frames awaiting annotation

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
    def update_detection_results(self, detections, frame_shape, track_history, tracked_objects_info):
        with self.detections_data_lock:
            self.latest_detections_data["results"] = detections
            self.latest_detections_data["frame_shape"] = frame_shape
            self.latest_detections_data["track_history"] = track_history
            self.latest_detections_data["tracked_objects_info"] = tracked_objects_info

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
            return self.latest_detections_data["tracked_objects_info"].copy()

    def get_track_history(self):
        with self.detections_data_lock:
            return self.latest_detections_data["track_history"].copy()

    # --- New Getter for Current Detections ---
    def get_current_detections_data(self):
        """Returns the latest raw detection results, frame shape, and track history."""
        with self.detections_data_lock:
            detections = self.latest_detections_data.get("results", [])
            shape = self.latest_detections_data.get("frame_shape")
            track_history_deques = self.latest_detections_data.get("track_history", {}) # Get track history (deques)

            frame_width = None
            frame_height = None
            if shape is not None and isinstance(shape, (tuple, list)) and len(shape) >= 2:
                frame_height = shape[0]
                frame_width = shape[1]

            # Convert deques to lists for JSON serialization
            track_history_lists = {
                track_id: list(points)
                for track_id, points in track_history_deques.items()
            }

            # Return the necessary parts for client-side drawing
            return {
                "detections": detections,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "track_history": track_history_lists # Return lists instead of deques
            }
    # -----------------------------------------

    # --- Lifecycle Management ---

    def start(self):
        if self.frame_grabber or self.object_detector or self.annotation_worker:
            logger.warning("Detection system already running or not properly stopped.")
            return

        logger.info("Starting DetectionSystem threads...")
        self.stop_event.clear()

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

        # tracked data cleared via resetting latest_detections_data

        # Instantiate workers, passing necessary dependencies and callbacks
        self.frame_grabber = FrameGrabber(
            stream_url=self.rtsp_stream_url,
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
            model_names=self.model.names
        )
        self.annotation_worker = AnnotationWorker(
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            annotated_frame_callback=self.update_latest_annotated_frame,
            max_track_points=self.max_track_points,
            model_names=self.model.names
        )

        # Start threads
        self.frame_grabber.start()
        time.sleep(0.1)
        self.object_detector.start()
        time.sleep(0.1)
        self.annotation_worker.start()

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

        logger.info("DetectionSystem threads stopped.")

    def is_running(self):
        """Check if worker threads are alive."""
        grabber_alive = self.frame_grabber and self.frame_grabber.is_alive()
        detector_alive = self.object_detector and self.object_detector.is_alive()
        annotator_alive = self.annotation_worker and self.annotation_worker.is_alive()
        return grabber_alive and detector_alive and annotator_alive

