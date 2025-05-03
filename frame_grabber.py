import cv2
import time
import threading
import queue
import os
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class FrameGrabber:
    def __init__(self, stream_url, frame_queue, stop_event, frame_update_callback):
        self.stream_url = stream_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.frame_update_callback = frame_update_callback # Callback to update central state
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _capture_frames(self):
        logger.info(f"FrameGrabber: THREAD STARTED. Capturing from {self.stream_url}")
        cap = None # Initialize cap to None
        try:
            # --- Try forcing TCP transport for RTSP ---
            logger.info("Attempting to force TCP transport for RTSP.")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            # -----------------------------------------

            logger.debug(f"Attempting to open stream with FFMPEG backend: {self.stream_url}")
            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.error(f"Error: Could not open video stream {self.stream_url}")
                self.stop_event.set() # Signal other threads to stop if capture fails
                return
            else:
                logger.info("Successfully opened stream with FFMPEG backend.")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Warning: Failed to grab frame from {self.stream_url}")
                    # Optional: Attempt to reopen the stream after a delay
                    time.sleep(1.0)
                    if cap:
                        cap.release()
                    cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        logger.error(f"Error: Failed to reopen video stream {self.stream_url}. Stopping grabber.")
                        self.stop_event.set()
                        break
                    else:
                        logger.info("Successfully reopened stream.")
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
            # --- Clean up environment variable ---
            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
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
