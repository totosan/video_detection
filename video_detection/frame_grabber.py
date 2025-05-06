import cv2
import time
import threading
import queue
import os
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class FrameGrabber:
    def __init__(self, source_identifier, source_type, frame_queue, stop_event, frame_update_callback):
        self.source_identifier = source_identifier
        self.source_type = source_type # "rtsp" or "device"
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_update_callback = frame_update_callback # Callback to update central state
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self):
        logger.info(f"FrameGrabber: THREAD STARTED. Capturing from {self.source_type} source: {self.source_identifier}")
        cap = None # Initialize cap to None
        try:
            if self.source_type == "rtsp":
                # --- Try forcing TCP transport for RTSP ---
                logger.info("Attempting to force TCP transport for RTSP.")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                # -----------------------------------------
                logger.debug(f"Attempting to open RTSP stream with FFMPEG backend: {self.source_identifier}")
                cap = cv2.VideoCapture(self.source_identifier, cv2.CAP_FFMPEG)
            elif self.source_type == "device":
                logger.debug(f"Attempting to open video device: {self.source_identifier}")
                cap = cv2.VideoCapture(self.source_identifier) # Default backend for local devices
            else:
                logger.error(f"Unsupported source_type: {self.source_type}. Stopping grabber.")
                self.stop_event.set()
                return

            if not cap.isOpened():
                logger.error(f"Error: Could not open video source {self.source_identifier} (type: {self.source_type})")
                self.stop_event.set() # Signal other threads to stop if capture fails
                return
            else:
                logger.info(f"Successfully opened video source {self.source_identifier} (type: {self.source_type}).")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Warning: Failed to grab frame from {self.source_identifier}")
                    # Optional: Attempt to reopen the stream after a delay
                    time.sleep(1.0)
                    if cap:
                        cap.release()
                    
                    # Reopen logic based on source type
                    if self.source_type == "rtsp":
                        cap = cv2.VideoCapture(self.source_identifier, cv2.CAP_FFMPEG)
                    elif self.source_type == "device":
                        cap = cv2.VideoCapture(self.source_identifier)
                    else: # Should not happen if initial check passed
                        logger.error("Cannot reopen, unknown source type during retry.")
                        self.stop_event.set()
                        break

                    if not cap.isOpened():
                        logger.error(f"Error: Failed to reopen video source {self.source_identifier}. Stopping grabber.")
                        self.stop_event.set()
                        break
                    else:
                        logger.info(f"Successfully reopened video source {self.source_identifier}.")
                    continue

                # Update the latest frame using the callback
                self.frame_update_callback(frame.copy())
                # logger.debug("FrameGrabber: called frame_update_callback") # Optional: can be noisy

                # Put frame in queue for detection (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    logger.debug("Warning: Detection frame queue full, discarding frame.")
                    pass

                time.sleep(0.01) # Small sleep to prevent busy-waiting

        except Exception as e:
            logger.exception("FrameGrabber: EXCEPTION in thread")
            self.stop_event.set() # Signal stop on unexpected error
        finally:
            # --- Clean up environment variable if it was set ---
            if self.source_type == "rtsp" and "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            # ------------------------------------
            if cap and cap.isOpened():
                cap.release()
            logger.info("FrameGrabber: THREAD EXITING")

    def stop(self):
        logger.debug("FrameGrabber stop called.")
        # Stop event should be set by the manager (DetectionSystem)
        if self.thread and self.thread.is_alive():
            logger.debug(f"Joining FrameGrabber thread (timeout=1s)... Thread ID: {self.thread.ident}")
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                logger.warning("Warning: Frame capture thread did not join cleanly.")
            else:
                logger.debug("FrameGrabber thread joined successfully.")
        else:
            logger.debug("FrameGrabber thread was not running or already joined.")
        self.thread = None # Clear thread reference

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()
