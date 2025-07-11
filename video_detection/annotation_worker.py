import threading
import queue
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AnnotationWorker:
    def __init__(self, annotation_queue, stop_event, annotated_frame_callback, max_track_points, model_names, is_backend_annotation_enabled_func):
        self.annotation_queue = annotation_queue
        self.stop_event = stop_event
        self.annotated_frame_callback = annotated_frame_callback
        self.max_track_points = max_track_points
        self.model_names = model_names
        self.is_backend_annotation_enabled_func = is_backend_annotation_enabled_func
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._annotate_loop, daemon=True)
        self.thread.start()

    def _annotate_loop(self):
        logger.info("AnnotationWorker thread started.")
        while not self.stop_event.is_set():
            try:
                item = self.annotation_queue.get(timeout=0.1)
                if item is None:
                    continue

                current_detections, frame_shape, track_history, tracked_objects_info, frame = item

                if not self.is_backend_annotation_enabled_func():
                    self.annotation_queue.task_done()
                    continue

            except queue.Empty:
                continue
            try:
                annotated = frame.copy()
                # Draw track lines
                # Iterate over a copy of the dictionary to prevent runtime errors
                for track_id, points in list(track_history.items()):
                    if len(points) > 1:
                        color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)
                        pts = [(int(x), int(y)) for x, y in points]
                        for i in range(1, len(pts)):
                            cv2.line(annotated, pts[i-1], pts[i], color, 2)
                
                # First pass: Draw segmentation masks
                for det in current_detections:
                    try:
                        if isinstance(det['box'], (list, tuple)) and len(det['box']) == 4:
                            x1, y1, x2, y2 = map(int, det['box'])
                        elif hasattr(det['box'], 'tolist'):
                            x1, y1, x2, y2 = map(int, det['box'].tolist())
                        else:
                            logger.warning(f"Unexpected format for det['box']: {type(det['box'])}")
                            continue

                        track_id = det.get('track_id')
                        color = tuple(int(c) for c in det['color']) if 'color' in det and det['color'] is not None else (0, 255, 0)
                        
                        # Draw segmentation mask if available
                        has_mask = det.get('has_mask', False)
                        if has_mask and track_id is not None and track_id in tracked_objects_info:
                            if 'segmentation_mask' in tracked_objects_info[track_id]:
                                try:
                                    # Get mask from tracked_objects_info
                                    mask = tracked_objects_info[track_id]['segmentation_mask']
                                    
                                    # Apply the colored mask as a semi-transparent overlay
                                    alpha = 0.5  # Transparency factor
                                    
                                    # Create mask image of the right size
                                    h, w = frame.shape[:2]
                                    binary_mask = np.zeros((h, w), dtype=np.uint8)
                                    
                                    # Convert mask to binary image sized to the frame
                                    # The mask might be differently sized or formatted
                                    try:
                                        logger.debug(f"Processing mask with shape {mask.shape}, frame shape: {frame.shape[:2]}")
                                        
                                        # Special case for newer YOLO models where mask is in a different format
                                        if len(mask.shape) == 2 and mask.shape[0] > 0 and mask.shape[1] == 2:
                                            # This is likely points format, convert to binary mask
                                            logger.debug("Converting points mask to binary mask")
                                            points = mask.astype(np.int32)
                                            binary_mask = np.zeros((h, w), dtype=np.uint8)
                                            cv2.fillPoly(binary_mask, [points], 1)
                                        elif mask.shape[:2] == (h, w):
                                            # If mask is already the right size
                                            logger.debug("Using mask as-is (same dimensions as frame)")
                                            binary_mask = (mask > 0.5).astype(np.uint8)
                                        else:
                                            # If mask needs to be resized or is in a different format
                                            logger.debug("Resizing mask to match frame dimensions")
                                            resized_mask = cv2.resize(mask.astype(np.float32), (w, h))
                                            binary_mask = (resized_mask > 0.5).astype(np.uint8)
                                    except Exception as e:
                                        logger.warning(f"Error processing mask: {e}")
                                        # Fall back to box-based mask
                                        logger.debug("Falling back to bounding box mask")
                                        binary_mask[y1:y2, x1:x2] = 1
                                    
                                    # Create a colored version of the mask
                                    colored_mask = np.zeros_like(annotated)
                                    colored_mask[binary_mask > 0] = color
                                    
                                    # Apply the overlay with transparency
                                    mask_indices = binary_mask > 0
                                    annotated[mask_indices] = cv2.addWeighted(
                                        colored_mask, alpha, annotated, 1 - alpha, 0
                                    )[mask_indices]
                                    
                                    # Add outline around the mask
                                    contours, _ = cv2.findContours(
                                        binary_mask, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE
                                    )
                                    cv2.drawContours(annotated, contours, -1, color, 2)
                                    
                                    logger.debug(f"Drew segmentation mask for track ID {track_id}")
                                except Exception as e:
                                    logger.warning(f"Error drawing segmentation mask: {e}")
                                    # Fall back to bounding box
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            else:
                                # Fall back to bounding box if mask is not found
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        else:
                            # Draw bounding box if no mask
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    except Exception as e:
                        logger.exception(f"Error processing mask: {e}")
                
                text_size = 2.5 # Adjusted for better fit
                # Second pass: Draw labels
                for det in current_detections:
                    try:
                        if isinstance(det['box'], (list, tuple)) and len(det['box']) == 4:
                            x1, y1, x2, y2 = map(int, det['box'])
                        elif hasattr(det['box'], 'tolist'):
                            x1, y1, x2, y2 = map(int, det['box'].tolist())
                        else:
                            continue

                        label = det['label']
                        track_id = det.get('track_id')
                        color = tuple(int(c) for c in det['color']) if 'color' in det and det['color'] is not None else (0, 255, 0)
                        
                        label_text = f"{label} (ID: {track_id})" if track_id is not None else label

                        # --- Text Placement Logic ---
                        text_x, text_y = 0, 0
                        text_origin_set = False

                        # Try to find center of segmentation mask first
                        if det.get('has_mask') and track_id is not None and track_id in tracked_objects_info and 'segmentation_mask' in tracked_objects_info[track_id]:
                            try:
                                mask = tracked_objects_info[track_id]['segmentation_mask']
                                h, w = frame.shape[:2]
                                # Convert to a binary mask of the correct size to find contours
                                if len(mask.shape) == 2 and mask.shape[0] > 0 and mask.shape[1] == 2:
                                    points = mask.astype(np.int32)
                                    binary_mask = np.zeros((h, w), dtype=np.uint8)
                                    cv2.fillPoly(binary_mask, [points], 1)
                                elif mask.shape[:2] == (h,w):
                                    binary_mask = (mask > 0.5).astype(np.uint8)
                                else:
                                    resized_mask = cv2.resize(mask.astype(np.float32), (w, h))
                                    binary_mask = (resized_mask > 0.5).astype(np.uint8)

                                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    # Find the largest contour and its center
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    M = cv2.moments(largest_contour)
                                    if M["m00"] > 0:
                                        text_x = int(M["m10"] / M["m00"])
                                        text_y = int(M["m01"] / M["m00"])
                                        text_origin_set = True
                            except Exception as e:
                                logger.warning(f"Could not calculate mask centroid for track {track_id}: {e}")

                        # Fallback to bounding box center if mask center fails or is not available
                        if not text_origin_set:
                            text_x = (x1 + x2) // 2
                            text_y = (y1 + y2) // 2
                        
                        # --- Draw Text with Background ---
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
                        
                        # Center the text block on the calculated origin (text_x, text_y)
                        rect_x1 = text_x - text_w // 2
                        rect_y1 = text_y - text_h // 2
                        rect_x2 = text_x + text_w // 2
                        rect_y2 = text_y + text_h // 2

                        cv2.rectangle(
                            annotated,
                            (rect_x1 - 2, rect_y1 - 2),
                            (rect_x2 + 2, rect_y2 + 4),
                            (0, 0, 0),  # black background
                            -1
                        )
                        cv2.putText(
                            annotated,
                            label_text,
                            (rect_x1, rect_y2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_size,
                            (255, 255, 255),  # white text
                            1
                        )

                    except Exception as e:
                        logger.exception(f"Error processing detection box: {e}")
                self.annotated_frame_callback(annotated)
            except Exception:
                logger.exception("Exception in AnnotationWorker during annotation")
            finally:
                self.annotation_queue.task_done()
        logger.info("AnnotationWorker thread stopped.")

    def stop(self):
        logger.debug("AnnotationWorker stop called.")
        if self.thread and self.thread.is_alive():
            logger.debug(f"Joining AnnotationWorker thread (timeout=2s)... Thread ID: {self.thread.ident}")
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                logger.warning("Warning: AnnotationWorker thread did not join cleanly.")
            else:
                logger.debug("AnnotationWorker thread joined successfully.")
        else:
            logger.debug("AnnotationWorker thread was not running or already joined.")
        self.thread = None

    def join(self, timeout=None):
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def is_alive(self):
        return self.thread and self.thread.is_alive()