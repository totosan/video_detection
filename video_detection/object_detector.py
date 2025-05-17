import time
import cv2
import queue
import math
import threading
from collections import deque
import logging # Import logging
import platform # Import platform module

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Configuration ---
FRAME_SKIP_FACTOR = 0 # Process every Nth frame (e.g., 2 means process 1, skip 1, process 1, ...)
# -------------------

class ObjectDetector:
    def __init__(self, model, frame_queue, annotation_queue, stop_event, results_update_callback, max_track_points, model_names, tracker_config_path=None): # Add tracker_config_path
        self.model = model
        self.frame_queue = frame_queue
        self.annotation_queue = annotation_queue
        self.stop_event = stop_event
        self.results_update_callback = results_update_callback
        self.max_track_points = max_track_points
        self.model_names = model_names
        self.tracker_config_path = tracker_config_path # Store the path
        self.thread = None
        # Determine device based on OS
        if platform.system() == "Darwin": # macOS
            self.device = 'mps'
        elif platform.system() == "Linux": # Assuming Linux is Jetson/other CUDA-capable
            # You might want more specific checks for CUDA availability here
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda:0' # Or 'cuda'
            else:
                self.device = 'cpu'
            # For simplicity, assuming CUDA is available on Linux for now
            # self.device = 'cuda:0' # Or 'cuda'  # This line is now redundant
        else:
            self.device = 'cpu' # Fallback to CPU
        logger.info(f"Using device: {self.device}")
        # Keep track history and info local to the detection process within the thread loop
        # They will be passed back via the callback

    def start(self):
        self.thread = threading.Thread(target=self._detect_objects, daemon=True)
        self.thread.start()

    def _detect_objects(self):
        logger.info("Detection thread started.")
        # Initialize tracking data here, it will be passed back via callback
        track_history = {}  # {track_id: deque(points)}
        tracked_objects_info = {}  # {track_id: {'name': str, 'last_seen': float, ...}}
        frame_counter = 0  # Initialize frame counter

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_counter += 1

                # --- Skip Frame Logic ---
                if FRAME_SKIP_FACTOR > 1 and frame_counter % FRAME_SKIP_FACTOR != 1:
                    logger.debug(f"Skipping frame {frame_counter} due to FRAME_SKIP_FACTOR.")
                    self.frame_queue.task_done()
                    continue  # Skip to the next frame
                # ------------------------

                # Process the frame using process_frame
                try:
                    results, frame_shape = self.process_frame(
                        frame, track_history, tracked_objects_info
                    )

                    # --- Convert results to serializable detections for annotation ---
                    serializable_detections = []
                    if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        clss = results[0].boxes.cls.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy() if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None else [None]*len(boxes)
                        for i in range(len(boxes)):
                            box = [int(x) for x in boxes[i]]
                            cls_idx = int(clss[i])
                            label = self.model.names[cls_idx] if cls_idx < len(self.model.names) else 'unknown'
                            color = ((track_ids[i] * 50) % 255, (track_ids[i] * 80) % 255, (track_ids[i] * 120) % 255) if track_ids[i] is not None else (255,0,0)
                            serializable_detections.append({
                                'box': box,
                                'label': label,
                                'color': color,
                                'track_id': int(track_ids[i]) if track_ids[i] is not None else None
                            })
                    # -------------------------------------------------------------

                    # Enqueue serializable detection data and frame for annotation
                    if serializable_detections:
                        try:
                            self.annotation_queue.put_nowait((
                                serializable_detections,
                                frame_shape,
                                track_history,
                                tracked_objects_info,
                                frame.copy()  # Send the frame that was actually processed
                            ))
                        except queue.Full:
                            logger.warning("Annotation queue is full; dropping frame annotation task.")

                    # Update central state with raw detection data from the processed frame
                    self.results_update_callback(
                        results,
                        frame_shape,
                        track_history,
                        tracked_objects_info
                    )
                except Exception as e:
                    logger.exception(f"Error processing frame {frame_counter}: {e}")
                    # Invoke callback with default data to avoid blocking
                    self.results_update_callback(None, None, {}, {})

                # Mark the task from frame_queue as done
                self.frame_queue.task_done()

        except Exception as e:
            logger.exception("ObjectDetector: EXCEPTION in thread")
            self.stop_event.set()  # Signal stop on error
        finally:
            logger.info("Detection thread stopped.")

    def stop(self):
        logger.debug("ObjectDetector stop called.")
        # Stop event should be set by the manager (DetectionSystem)
        if self.thread and self.thread.is_alive():
            logger.debug(f"Joining ObjectDetector thread (timeout=2s)... Thread ID: {self.thread.ident}")
            # Ensure the queue processing can finish if blocked on get()
            # Putting None might be needed if timeout isn't sufficient
            # try: self.frame_queue.put_nowait(None) # Sentinel value if needed
            # except queue.Full:
            #     logger.warning("Could not put sentinel in full frame queue during stop.")

            self.thread.join(timeout=2) # Allow time for detection cycle
            if self.thread.is_alive():
                logger.warning("Warning: Detection thread did not join cleanly.")
            else:
                logger.debug("ObjectDetector thread joined successfully.")
        else:
            logger.debug("ObjectDetector thread was not running or already joined.")
        self.thread = None # Clear thread reference

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def process_frame(self, frame, track_history, tracked_objects_info):
        """Process a video frame and return detection results."""
        if frame is None:
            logger.warning("Frame is None, skipping processing.")
            return None, None

        # Perform detection using the determined device
        track_args = {
            "source": frame,
            "persist": True,  # Important for tracking across skipped frames
            "verbose": False,
            "conf": 0.25,
            "vid_stride":3,
            "device": self.device
        }
        if self.tracker_config_path:
            track_args["tracker"] = self.tracker_config_path

        results = self.model.track(**track_args)
        frame_shape = frame.shape[:2]  # Store height, width of the processed frame

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            track_ids = None

            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()

                # Extract tracked detections
                for box, conf, cl, track_id in zip(boxes, confs, cls, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_id)

                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    # Generate color based on track_id
                    color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)

                    # Only store if region is valid
                    if x2 > x1 and y2 > y1:
                        try:
                            # Make a copy of the region to avoid reference issues
                            detection_region = frame[y1:y2, x1:x2].copy()
                            # Add detection image to tracked_objects_info
                            tracked_objects_info[track_id] = tracked_objects_info.get(track_id, {})
                            tracked_objects_info[track_id]['detection_image'] = detection_region
                            logger.debug(f"✅ Stored image for track ID {track_id}, region shape: {detection_region.shape}")
                        except Exception as e:
                            logger.exception(f"Error storing image for track ID {track_id}: {e}")
                    else:
                        logger.warning(f"Invalid bounding box for track ID {track_id}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

                    # Update track history
                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=self.max_track_points)
                    track_history[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))

                    # Retrieve class name
                    class_name = self.model.names[int(cl)] if int(cl) < len(self.model.names) else "unknown"
                    logger.debug(f"Assigned class name '{class_name}' for track ID {track_id}")

                    # Update tracked_objects_info
                    tracked_objects_info[track_id].update({
                        'name': class_name,
                        'last_seen': time.time(),
                        'last_box': (x1, y1, x2, y2),
                        'color': color
                    })

        return results, frame_shape
