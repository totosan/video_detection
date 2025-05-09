import threading
import queue
import cv2
import logging

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
                for track_id, points in track_history.items():
                    if len(points) > 1:
                        color = ((track_id * 50) % 255, (track_id * 80) % 255, (track_id * 120) % 255)
                        pts = [(int(x), int(y)) for x, y in points]
                        for i in range(1, len(pts)):
                            cv2.line(annotated, pts[i-1], pts[i], color, 2)
                # Draw detection boxes and labels
                for det in current_detections:
                    try:
                        # Ensure det['box'] is in the expected format and all values are ints
                        if isinstance(det['box'], (list, tuple)) and len(det['box']) == 4:
                            x1, y1, x2, y2 = map(int, det['box'])
                        elif hasattr(det['box'], 'tolist'):
                            x1, y1, x2, y2 = map(int, det['box'].tolist())
                        else:
                            logger.warning(f"Unexpected format for det['box']: {type(det['box'])}")
                            continue

                        label = det['label']
                        # Ensure color is a tuple of ints
                        color = tuple(int(c) for c in det['color']) if 'color' in det and det['color'] is not None else (0, 255, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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