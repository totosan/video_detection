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

        # --- Mode Configuration ---
        self.single_image_mode = not bool(self.rtsp_stream_url_config)
        if self.single_image_mode:
            logger.info("No RTSP_STREAM_URL configured. DetectionSystem will operate in single image mode.")

        # --- Determine Source Type and Identifier (only if not in single image mode) ---
        self.source_identifier = None
        self.source_type = None # "rtsp" or "device"
        if not self.single_image_mode:
            self._set_source_from_config(self.rtsp_stream_url_config) # Use a helper
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

    def _set_source_from_config(self, source_config_value):
        """Helper to set source_identifier and source_type based on a config string or int."""
        logger.info(f"_set_source_from_config: Received value: {source_config_value!r} (type: {type(source_config_value)})")
        # Normalize input: if string, strip whitespace
        if isinstance(source_config_value, str):
            source_config_value = source_config_value.strip()
        # Try to interpret as device index if possible
        if isinstance(source_config_value, int) or (isinstance(source_config_value, str) and source_config_value.isdigit()):
            try:
                self.source_identifier = int(source_config_value)
                self.source_type = "device"
                logger.info(f"Video source set to device index: {self.source_identifier}")
            except ValueError:
                logger.error(f"Could not parse '{source_config_value}' as a device index. Defaulting to RTSP/URL interpretation.")
                self.source_identifier = str(source_config_value)
                self.source_type = "rtsp"
        elif isinstance(source_config_value, str) and not source_config_value:
            logger.error("Video source string is empty. Cannot set source.")
            raise ValueError("Video source string cannot be empty.")
        else:
            self.source_identifier = str(source_config_value)
            self.source_type = "rtsp"
            logger.info(f"Video source set to RTSP/URL: {self.source_identifier}")

    def change_video_source(self, new_source_identifier):
        """Stops the current video stream, changes the source, and restarts."""
        logger.info(f"Attempting to change video source to: {new_source_identifier}")
        try:
            # Stop existing workers
            self.stop() # This should wait for threads to join (with timeout)
            logger.info("Detection system stopped for source change.")

            # Update the source identifier and type
            self._set_source_from_config(new_source_identifier)

            # Restart the system with the new source
            # The start() method will use the updated self.source_identifier and self.source_type
            self.start() # This will now be more synchronous and will raise RuntimeError on failure
            logger.info(f"Detection system restarted with new source: {self.source_identifier}")
            return True, f"Successfully changed video source to {self.source_identifier}"
        except RuntimeError as e: # Catch specific error from start()
            logger.error(f"Failed to start detection system with new source {new_source_identifier}: {e}")
            # Attempt to stop again to ensure clean state if start failed partially
            self.stop_event.set() # Signal any potentially lingering threads from the failed start
            # Ensure worker instances are None so a subsequent start attempt is clean
            self.frame_grabber = None
            self.object_detector = None
            self.annotation_worker = None
            return False, f"Error starting system with new video source: {str(e)}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred while changing video source to {new_source_identifier}")
            self.stop_event.set() # Ensure stop signal is set
            # Ensure worker instances are None
            self.frame_grabber = None
            self.object_detector = None
            self.annotation_worker = None
            return False, f"Unexpected error changing video source: {str(e)}"

    # --- State Update Callbacks/Methods ---

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
        if self.single_image_mode:
            logger.info("DetectionSystem is in single image mode. Workers will not be started.")
            # Ensure model is loaded, as it's needed for process_single_image
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not loaded in single image mode. This should not happen if __init__ completed.")
                raise RuntimeError("Model not loaded, cannot operate in single image mode.")
            return # Do not start workers

        if self.frame_grabber or self.object_detector or self.annotation_worker:
            logger.warning("Detection system already running or not properly stopped. Attempting to stop first.")
            self.stop() # Ensure a clean stop before trying to start again.

        logger.info("Starting DetectionSystem threads...")
        self.stop_event.clear()

        # Clear queues and reset state SYNCHRONOUSLY
        self._clear_queues_and_reset_state()
        logger.info("Queues and state cleared.")


        # Instantiate workers, passing necessary dependencies and callbacks
        logger.info(f"Initializing FrameGrabber for source: {self.source_identifier} (type: {self.source_type})")
        self.frame_grabber = FrameGrabber(
            source_identifier=self.source_identifier,
            source_type=self.source_type,
            frame_queue=self.frame_queue,
            stop_event=self.stop_event,
            frame_update_callback=self.update_latest_frame
        )
        logger.info("Initializing ObjectDetector...")
        # Ensure model.names is available, provide an empty list or default if not
        model_names = self.model.names if hasattr(self.model, 'names') and self.model.names is not None else []

        self.object_detector = ObjectDetector(
            model=self.model,
            frame_queue=self.frame_queue,
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            results_update_callback=self.update_detection_results,
            max_track_points=self.max_track_points,
            model_names=model_names,
            tracker_config_path=CUSTOM_TRACKER_CONFIG
        )
        logger.info("Initializing AnnotationWorker...")
        self.annotation_worker = AnnotationWorker(
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            annotated_frame_callback=self.update_latest_annotated_frame,
            max_track_points=self.max_track_points,
            model_names=model_names,
            is_backend_annotation_enabled_func=self.is_backend_annotation_enabled
        )
        logger.info("Worker instances created.")

        # Start threads SYNCHRONOUSLY (the method itself will manage starting threads)
        self._start_workers()
        # _start_workers will log success or failure of thread starts and raise error if needed.


    def _clear_queues_and_reset_state(self):
        logger.info("Clearing queues and resetting state...")
        # Clear queues
        while not self.frame_queue.empty():
            try:
                item = self.frame_queue.get_nowait()
                self.frame_queue.task_done() # Ensure task_done is called
            except queue.Empty:
                break
            except Exception as e:
                logger.warning(f"Error getting from frame_queue during clear: {e}")
                break # Avoid infinite loop on other errors

        while not self.annotation_queue.empty():
            try:
                item = self.annotation_queue.get_nowait()
                self.annotation_queue.task_done() # Ensure task_done is called
            except queue.Empty:
                break
            except Exception as e:
                logger.warning(f"Error getting from annotation_queue during clear: {e}")
                break
        
        # Reset state variables
        with self.frame_lock:
            self.latest_frame = None
        with self.annotated_frame_lock:
            self.latest_annotated_frame = None
        with self.detections_data_lock:
            self.latest_detections_data = {"results": None, "frame_shape": None, "track_history": {}, "tracked_objects_info": {}}
        with self.detection_images_lock: # Also clear detection images
            self.detection_images.clear()
        logger.info("Queues and state reset complete.")


    def _start_workers(self):
        logger.info("Attempting to start worker threads...")

        logger.info("Starting FrameGrabber thread...")
        if self.frame_grabber:
            self.frame_grabber.start()
        time.sleep(0.05) 

        logger.info("Starting ObjectDetector thread...")
        if self.object_detector:
            self.object_detector.start()
        time.sleep(0.05)

        logger.info("Starting AnnotationWorker thread...")
        if self.annotation_worker:
            self.annotation_worker.start()
        time.sleep(0.05)

        # Check if threads actually started
        fg_alive = self.frame_grabber and self.frame_grabber.is_alive()
        od_alive = self.object_detector and self.object_detector.is_alive()
        aw_alive = self.annotation_worker and self.annotation_worker.is_alive()

        if fg_alive and od_alive and aw_alive:
            logger.info("All DetectionSystem worker threads appear to have started successfully.")
        else:
            logger.error(f"One or more DetectionSystem threads failed to start. Status: FG_alive={fg_alive}, OD_alive={od_alive}, AW_alive={aw_alive}")
            self.stop_event.set() # Signal all to stop
            # Raise an error to indicate failure to start, which can be caught by change_video_source
            raise RuntimeError(f"Failed to start all worker threads. Status: FG_alive={fg_alive}, OD_alive={od_alive}, AW_alive={aw_alive}")

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

    def toggle_tracking_and_bounding_boxes(self):
        """Toggle the state of tracking and bounding box drawing."""
        if not hasattr(self, 'draw_tracking_and_bounding_boxes'):
            self.draw_tracking_and_bounding_boxes = True  # Initialize if not present
        self.draw_tracking_and_bounding_boxes = not self.draw_tracking_and_bounding_boxes
        return self.draw_tracking_and_bounding_boxes

    def is_tracking_and_bounding_boxes_enabled(self):
        """Check if tracking and bounding box drawing is enabled."""
        return getattr(self, 'draw_tracking_and_bounding_boxes', True)

    def set_object_filter(self, object_filter):
        """Set the object filter for displaying specific labels."""
        if not hasattr(self, 'object_filter'):
            self.object_filter = None  # Initialize if not present
        self.object_filter = object_filter

    def get_object_filter(self):
        """Get the current object filter."""
        return getattr(self, 'object_filter', None)

    def process_single_image(self, image_np):
        """
        Processes a single image for object detection.

        Args:
            image_np (np.ndarray): The image to process (OpenCV format, BGR).

        Returns:
            list: A list of detection dictionaries, e.g.,
                  [{'box': [x_min, y_min, x_max, y_max], 'label': 'person', 'confidence': 0.9}, ...]
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.error("YOLO model not loaded. Cannot process single image.")
            return []

        logger.info(f"Processing single image of shape {image_np.shape}")
        try:
            # Perform detection
            # The results object from YOLO might vary slightly depending on the task (detect, track, etc.)
            # For simple detection, it's usually a list of Results objects.
            results = self.model.predict(source=image_np, verbose=False) # verbose=False to reduce console output

            serializable_detections = []
            if results and isinstance(results, list): # results is a list of Results objects
                for res in results: # Iterate through each Results object (usually one for a single image)
                    if hasattr(res, "boxes") and res.boxes is not None:
                        boxes = res.boxes.xyxyn.cpu().numpy()  # Normalized [x_min, y_min, x_max, y_max]
                        confs = res.boxes.conf.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy()
                        
                        img_height, img_width = image_np.shape[:2]

                        for i in range(len(boxes)):
                            box_normalized = boxes[i]
                            # Denormalize box coordinates
                            x_min = float(box_normalized[0] * img_width)
                            y_min = float(box_normalized[1] * img_height)
                            x_max = float(box_normalized[2] * img_width)
                            y_max = float(box_normalized[3] * img_height)
                            
                            conf = float(confs[i])
                            cls_idx = int(clss[i])
                            label = self.model.names[cls_idx] if cls_idx < len(self.model.names) else "unknown"

                            serializable_detections.append({
                                "box": [x_min, y_min, x_max, y_max], # Standard [x_min, y_min, x_max, y_max]
                                "label": label,
                                "confidence": conf
                            })
            logger.info(f"Detected {len(serializable_detections)} objects in the single image.")
            return serializable_detections
        except Exception as e:
            logger.exception("Error during single image processing")
            return []

