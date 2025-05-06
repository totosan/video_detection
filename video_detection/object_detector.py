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
FRAME_SKIP_FACTOR = 2 # Process every Nth frame (e.g., 2 means process 1, skip 1, process 1, ...)
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
        track_history = {} # {track_id: deque(points)}
        tracked_objects_info = {} # {track_id: {'name': str, 'last_seen': float, ...}}
        frame_counter = 0 # Initialize frame counter

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_counter += 1

                # --- Skip Frame Logic ---
                if FRAME_SKIP_FACTOR > 1 and frame_counter % FRAME_SKIP_FACTOR != 1:
                    # Skip processing this frame, but mark task as done
                    self.frame_queue.task_done()
                    # Optional: Add a small sleep to yield CPU if needed,
                    # but usually the queue.get timeout handles this.
                    # time.sleep(0.001)
                    continue # Skip to the next frame
                # ------------------------

                # --- Time the inference ---
                start_time = time.perf_counter()
                # Perform detection using the determined device
                # Pass the tracker config if provided
                track_args = {
                    "source": frame,
                    "persist": True, # Important for tracking across skipped frames
                    "verbose": False,
                    "conf": 0.25,
                    "device": self.device
                }
                if self.tracker_config_path:
                    track_args["tracker"] = self.tracker_config_path

                results = self.model.track(**track_args)
                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000
                logger.debug(f"Inference time: {inference_time_ms:.2f} ms on frame {frame_counter}")
                # --------------------------

                current_detections = [] # Store detections for this specific frame
                frame_shape = frame.shape[:2] # Store height, width of the processed frame

                # Process results (only if the frame wasn't skipped)
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    cls = results[0].boxes.cls.cpu().numpy()
                    track_ids = None
                    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id.cpu().numpy()
                        current_frame_track_ids = set() # Keep track of IDs seen in this frame

                        # Extract tracked detections
                        for box, conf, cl, track_id in zip(boxes, confs, cls, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            track_id = int(track_id)
                            current_frame_track_ids.add(track_id)
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            label = f"{self.model.names[int(cl)]} #{track_id} {conf:.2f}"
                            # Generate color based on track_id for consistency
                            color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)

                            # Store detection info
                            current_detections.append({
                                'box': (x1, y1, x2, y2),
                                'label': label,
                                'color': color,
                                'track_id': track_id,
                                'center': center
                            })

                            # Update track history
                            if track_id not in track_history:
                                track_history[track_id] = deque(maxlen=self.max_track_points)
                            track_history[track_id].append(center)

                            # Update tracked_objects_info
                            tracked_objects_info[track_id] = {
                                'name': self.model.names[int(cl)],
                                'last_seen': time.time(), # Use current time
                                'last_box': (x1, y1, x2, y2)
                            }
                        # Note: Track cleanup logic might need adjustment if skipping many frames.
                        # The 'persist=True' helps YOLO maintain tracks, but very large skips
                        # might still cause tracks to be lost or reassigned incorrectly.

                    else:
                        # Extract non-tracked detections
                        for box, conf, cl in zip(boxes, confs, cls):
                            x1, y1, x2, y2 = map(int, box)
                            label = f"{self.model.names[int(cl)]} {conf:.2f}"
                            # Store detection info
                            current_detections.append({
                                'box': (x1, y1, x2, y2),
                                'label': label,
                                'color': (0, 255, 0), # Default color
                                'track_id': None,
                                'center': None
                            })

                # Enqueue raw detection data and frame for annotation
                # Important: Enqueue even if no detections, but only if frame was processed
                try:
                    self.annotation_queue.put_nowait((
                        current_detections,
                        frame_shape,
                        track_history,
                        tracked_objects_info,
                        frame.copy() # Send the frame that was actually processed
                    ))
                except queue.Full:
                    logger.warning("Annotation queue is full; dropping frame annotation task.")


                # Update central state with raw detection data from the processed frame
                self.results_update_callback(
                    current_detections,
                    frame_shape,
                    track_history,
                    tracked_objects_info
                )
                # logger.debug("ObjectDetector: called results_update_callback") # Optional: can be noisy

                # Mark the task from frame_queue as done
                self.frame_queue.task_done()

        except Exception as e:
            logger.exception("ObjectDetector: EXCEPTION in thread")
            self.stop_event.set() # Signal stop on error
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
