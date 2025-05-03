import cv2
import time
import threading
import queue
import os # Import os
import config  # to update shared latest_frame and use frame_lock
from config import stop_event, frame_queue  # keep stop_event, frame_queue imports
import logging # Import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class FrameGrabber:
    def __init__(self, stream_url, frame_queue, stop_event):
        self.stream_url = stream_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self):
        logger.info(f"FrameGrabber: THREAD STARTED. Capturing from {self.stream_url}") # Use info
        try:
            # --- Try forcing TCP transport for RTSP ---
            logger.info("Attempting to force TCP transport for RTSP.")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            # -----------------------------------------

            logger.debug(f"Attempting to open stream with FFMPEG backend: {self.stream_url}")
            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.error(f"Error: Could not open video stream {self.stream_url}") # Use error
                self.stop_event.set()
                return
            else:
                logger.info("Successfully opened stream with FFMPEG backend.") # Use info

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Warning: Failed to grab frame from {self.stream_url}") # Use warning
                    time.sleep(0.5)
                    continue

                # Update the latest frame for display
                with config.frame_lock:
                    config.latest_frame = frame.copy()
                    logger.debug("FrameGrabber: updated config.latest_frame") # Use debug

                # Put frame in queue for detection (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    logger.debug("Warning: Detection frame queue full, discarding frame.") # Use debug for queue full
                    pass

                time.sleep(0.01)

        except Exception as e:
            logger.exception("FrameGrabber: EXCEPTION in thread") # Use exception
        finally:
            # --- Clean up environment variable ---
            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            # ------------------------------------
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            logger.info("FrameGrabber: THREAD EXITING") # Use info

    def stop(self):
        # Ensure stop_event is set via app.py or main control logic
        # self.stop_event.set() # Setting is usually done externally
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
            if self.thread.is_alive(): # Corrected syntax: use .is_alive()
                logger.warning("Warning: Frame capture thread did not join cleanly.") # Use warning
