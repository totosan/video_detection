import time
import cv2
import queue
import math
import threading
from collections import deque
import logging # Import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model, frame_queue, annotation_queue, stop_event, results_update_callback, max_track_points, model_names):
        self.model = model
        self.frame_queue = frame_queue
        self.annotation_queue = annotation_queue
        self.stop_event = stop_event
        self.results_update_callback = results_update_callback
        self.max_track_points = max_track_points
        self.model_names = model_names
        self.thread = None
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

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Perform detection
                results = self.model.track(frame, persist=True, verbose=False)
                current_detections = [] # Store detections for this specific frame
                frame_shape = frame.shape[:2] # Store height, width of the processed frame

                # Enqueue raw detection data and frame for annotation
                try:
                    self.annotation_queue.put_nowait((
                        current_detections,
                        frame_shape,
                        track_history,
                        tracked_objects_info,
                        frame.copy()
                    ))
                except queue.Full:
                    logger.warning("Annotation queue is full; dropping frame annotation task.")

                # Process results
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
                                'last_seen': time.time(),
                                'last_box': (x1, y1, x2, y2)
                            }

                        # --- Clean up old tracks --- 
                        # Remove tracks that haven't been seen in this frame
                        # This is a simple approach; more robust methods exist (e.g., time-based expiry)
                        # Be careful modifying dict while iterating; create a list of keys to remove
                        # removed_ids = [tid for tid in track_history if tid not in current_frame_track_ids]
                        # for tid in removed_ids:
                        #     del track_history[tid]
                        #     if tid in tracked_objects_info:
                        #         del tracked_objects_info[tid]
                        # Consider a time-based cleanup in the DetectionSystem or here if needed
                        # ---------------------------

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

                # Update central state with raw detection data
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
