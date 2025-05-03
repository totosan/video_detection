import time
import cv2
import queue
import math
import threading
from collections import deque
import logging # Import logging
import config # Import config to access DEBUG flag
# Import shared variables, remove processed_queue import
from config import (
    model, frame_queue, stop_event, track_history,
    tracked_objects_info, max_track_points,
    latest_detections, detections_lock,
    latest_annotated_frame, annotated_frame_lock
)

# Get logger for this module
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self): # No queues needed in constructor anymore
        self.model = model
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.thread = None
        # Keep track history local to the instance if it's only updated here
        self.track_history = track_history # Or manage instance specific if needed
        self.tracked_objects_info = tracked_objects_info # Shared info dict
        self.max_track_points = max_track_points

    def start(self):
        self.thread = threading.Thread(target=self._detect_objects, daemon=True)
        self.thread.start()

    def _detect_objects(self):
        logger.info("Detection thread started.") # Use info
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

                # --- Draw annotations on a copy of the frame --- 
                annotated_frame = results[0].plot() # Use the built-in plot function
                # -----------------------------------------------

                # --- Store the annotated frame --- 
                with config.annotated_frame_lock:
                    config.latest_annotated_frame = annotated_frame
                    logger.debug("ObjectDetector: updated config.latest_annotated_frame")
                # ---------------------------------

                # Process results
                for result in results:
                    if result.boxes is None:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    track_ids = None
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy()

                    # Extract tracked detections
                    if track_ids is not None:
                        for box, conf, cl, track_id in zip(boxes, confs, cls, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            track_id = int(track_id)
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            label = f"{self.model.names[int(cl)]} #{track_id} {conf:.2f}"
                            # Generate color based on track_id for consistency
                            color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)

                            # Store detection info instead of drawing immediately
                            current_detections.append({
                                'box': (x1, y1, x2, y2),
                                'label': label,
                                'color': color,
                                'track_id': track_id,
                                'center': center # Keep center if needed for track history
                            })

                            # Update track history (using the shared track_history dict)
                            if track_id not in self.track_history:
                                self.track_history[track_id] = deque(maxlen=self.max_track_points)
                            self.track_history[track_id].append(center)

                            # Update tracked_objects_info (example)
                            self.tracked_objects_info[track_id] = {
                                'name': self.model.names[int(cl)],
                                'last_seen': time.time(),
                                'last_box': (x1, y1, x2, y2)
                            }

                    # Extract non-tracked detections
                    else:
                        for box, conf, cl in zip(boxes, confs, cls):
                            x1, y1, x2, y2 = map(int, box)
                            label = f"{self.model.names[int(cl)]} {conf:.2f}"
                            # Store detection info
                            current_detections.append({
                                'box': (x1, y1, x2, y2),
                                'label': label,
                                'color': (0, 255, 0), # Default color for non-tracked
                                'track_id': None,
                                'center': None
                            })

                # Update shared latest_detections structure
                with detections_lock:
                    # Use global scope for assignment within the lock
                    global latest_detections
                    latest_detections["results"] = current_detections
                    latest_detections["frame_shape"] = frame_shape
                    if config.DEBUG:
                        logger.debug(f"Detector: Updated latest_detections with {len(current_detections)} results.") # Use debug

                # Mark the task from frame_queue as done
                self.frame_queue.task_done()

        except Exception as e:
            logger.exception("ObjectDetector: EXCEPTION in thread") # Use exception
        finally:
            logger.info("Detection thread stopped.") # Use info

    def stop(self):
        # Ensure stop_event is set via app.py or main control logic
        # self.stop_event.set() # Setting is usually done externally
        if self.thread and self.thread.is_alive():
            # Attempt to signal queue processing to finish
            # self.frame_queue.put(None) # Sentinel value if needed, but timeout works
            self.thread.join(timeout=2) # Allow more time for detection to finish
            if self.thread.is_alive():
                logger.warning("Warning: Detection thread did not join cleanly.") # Use warning
