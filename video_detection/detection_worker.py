import threading
import queue
import logging
import time

# Get logger for this module
logger = logging.getLogger(__name__)

class DetectionWorker:
    """Worker for processing frames with object detection in a separate thread."""
    
    def __init__(self, detector, detection_system, frame_queue, stop_event):
        """Initialize the detection worker.
        
        Args:
            detector: The object detector to use for processing frames
            detection_system: The detection system to update with results
            frame_queue: Queue containing frames to process
            stop_event: Event to signal when to stop processing
        """
        self.detector = detector
        self.detection_system = detection_system
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.thread = None
        
    def start(self):
        """Start the detection worker thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            logger.info("Detection worker thread started.")
        else:
            logger.warning("Detection worker thread already running.")
            
    def stop(self):
        """Stop the detection worker thread."""
        if self.thread and self.thread.is_alive():
            logger.debug("Stopping detection worker thread...")
            # The stop_event should already be set by the detection system
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning("Detection worker thread did not join cleanly.")
            else:
                logger.debug("Detection worker thread stopped successfully.")
        self.thread = None
        
    def is_alive(self):
        """Check if the worker thread is alive."""
        return self.thread is not None and self.thread.is_alive()
        
    def run(self):
        """Main loop for detection worker."""
        logger.info("Detection worker running.")
        
        frame_count = 0
        
        while not self.stop_event.is_set():
            # Get next frame from queue
            try:
                frame = self.frame_queue.get(timeout=0.1)
                frame_count += 1
            except queue.Empty:
                continue
                
            # Process the frame
            try:
                if frame is not None:
                    # Updated to receive detection images
                    detections, frame_shape, track_history, tracked_objects_info, detection_images = self.detector.process_frame(frame)
                    
                    # Update the detection system with all data
                    self.detection_system.update_detection_results(
                        detections,
                        frame_shape,
                        track_history,
                        tracked_objects_info,
                        detection_images  # Pass the detection images dictionary
                    )
                    
                    if frame_count % 100 == 0:
                        logger.debug(f"Processed {frame_count} frames")
                        
                # Mark the frame as processed
                self.frame_queue.task_done()
            except Exception as e:
                logger.exception(f"Error processing frame {frame_count}: {e}")
                # Still mark as done to avoid blocking
                try:
                    self.frame_queue.task_done()
                except Exception:
                    pass
                
                # Brief pause to avoid tight loop if errors persist
                time.sleep(0.1)
                
        logger.info("Detection worker stopped.")