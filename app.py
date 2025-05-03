import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import time
import threading # Import threading
import numpy as np # Import numpy
from flask import Flask, render_template, Response, jsonify, request
from collections import deque # Import deque if needed for track history drawing
import logging # Import logging

# Import shared variables, remove processed_queue
import config
from config import (
    STATIC_FOLDER, TEMPLATE_FOLDER, stop_event, frame_queue,
    latest_frame, frame_lock, latest_detections, detections_lock,
    tracked_objects_info, latest_annotated_frame, annotated_frame_lock # Import new annotated frame vars
)
from frame_grabber import FrameGrabber
from object_detector import ObjectDetector

# Add the raw RTSP URL constant for direct capture
RAW_STREAM_URL = "..."

# Initialize Flask app
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Get the root logger (or create a specific one if preferred)
logger = logging.getLogger(__name__) # Use module name for logger

# Global references to thread workers
capture_grabber = None
detection_worker = None

def generate_mjpeg_stream():
    """Generates MJPEG stream by overlaying latest detections on latest frame."""
    logger.debug("MJPEG: Waiting for first frame...")
    # Wait until first frame is available
    while not stop_event.is_set():
        with config.frame_lock:
            if config.latest_frame is not None:
                break
        time.sleep(0.01)
    logger.debug("MJPEG: First frame received, starting stream loop.")

    frame_count = 0
    last_log_time = time.time()

    while not stop_event.is_set():
        # Get a snapshot of the latest frame
        with config.frame_lock:
            frame = config.latest_frame.copy()
        # Overlay detections
        with config.detections_lock:
            dets = config.latest_detections.get("results") or []
        for det in dets:
            x1, y1, x2, y2 = det['box']
            color = det['color']
            label = det['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            track_id = det.get('track_id')
            if track_id is not None and track_id in config.track_history:
                hist = config.track_history[track_id]
                for i in range(len(hist)-1):
                    cv2.line(frame, hist[i], hist[i+1], color, 2)
        # Encode and yield
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("MJPEG: Error encoding frame.") # Use error level
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

        # Log frame rate periodically if DEBUG is True
        frame_count += 1
        now = time.time()
        if config.DEBUG and (now - last_log_time >= 5.0):
            logger.debug(f"MJPEG: Yielded {frame_count} frames in the last 5 seconds.")
            frame_count = 0
            last_log_time = now

        time.sleep(0.03)
    logger.debug("MJPEG: Exited stream generation loop.")

def generate_frames(lock, frame_source_func):
    """Generator function to yield frames for streaming."""
    while True:
        time.sleep(0.03) # Limit frame rate slightly
        with lock:
            frame = frame_source_func()
            if frame is None:
                # Send a placeholder image if no frame is available
                # You might want to create a small black image or load a static image
                # For simplicity, we'll just skip if None for now
                continue

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logger.warning("Could not encode frame to JPEG") # Use warning
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.exception(f"Error encoding frame: {e}") # Use exception

@app.route('/')
def index():
    """Serves the main HTML page."""
    logger.debug("Serving index.html")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for the raw camera feed."""
    logger.info("Video feed requested.") # Use info
    return Response(generate_frames(config.frame_lock, lambda: config.latest_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_annotated')
def video_feed_annotated():
    """Video streaming route for the feed with detections."""
    logger.info("Annotated video feed requested.") # Use info
    return Response(generate_frames(config.annotated_frame_lock, lambda: config.latest_annotated_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/tracked_objects')
def api_tracked_objects():
    """API endpoint to get the latest tracked object information in the format expected by JS."""
    current_time = time.time()
    objects_list = []
    # Create a copy of the keys to avoid runtime errors if the dict changes during iteration
    current_tracked_ids = list(config.tracked_objects_info.keys())

    for track_id in current_tracked_ids:
        if track_id in config.tracked_objects_info: # Check if ID still exists
            info = config.tracked_objects_info[track_id]
            time_since_seen = round(current_time - info.get('last_seen', current_time), 1)
            objects_list.append({
                'id': track_id,
                'name': info.get('name', 'Unknown'),
                'last_seen_timestamp': info.get('last_seen', 0), # Send the raw timestamp
                'time_since_seen': time_since_seen
                # Add other fields from info if needed by JS
            })
        else:
             logger.info(f"API: Track ID {track_id} disappeared during list creation.")

    # Sort by time_since_seen (most recent first)
    objects_list.sort(key=lambda x: x['time_since_seen'])

    # print(f"API: Sending tracked objects list: {objects_list}") # Log the list being sent
    try:
        return jsonify(objects_list) # Return the list
    except Exception as e:
        logger.exception("API: Error converting tracked_objects list to JSON") # Use exception
        return jsonify({"error": "Failed to serialize tracked objects data"}), 500

@app.route('/snapshot')
def snapshot():
    """Returns a single JPEG snapshot from the latest captured frame."""
    # Ensure frame grabber is running
    if not capture_grabber or not capture_grabber.thread.is_alive():
        logger.warning("snapshot: FrameGrabber not alive, restarting threads.") # Use warning
        start_threads()

    logger.debug("Snapshot: Retrieving from latest_frame.")
    with config.frame_lock:
        if config.latest_frame is None:
            logger.info("Snapshot: No frame available.") # Use info
            return ("No frame available", 503)
        frame = config.latest_frame.copy()

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        logger.error("Snapshot: Error encoding frame.") # Use error
        return ("Error encoding frame", 500)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/raw_snapshot')
def raw_snapshot():
    """Returns a single JPEG snapshot directly from the RTSP stream (no threads)."""
    logger.debug("Raw snapshot: Opening direct RTSP capture.")
    cap = cv2.VideoCapture(RAW_STREAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Raw snapshot: Unable to open stream {RAW_STREAM_URL}") # Use error
        return ("Cannot open stream", 503)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logger.error("Raw snapshot: Failed to grab frame.") # Use error
        return ("Failed to grab frame", 503)
    ret2, buf = cv2.imencode('.jpg', frame)
    if not ret2:
        logger.error("Raw snapshot: Error encoding frame.") # Use error
        return ("Error encoding frame", 500)
    logger.debug("Raw snapshot: Returning JPEG image.")
    return Response(buf.tobytes(), mimetype='image/jpeg')

def start_threads():
    """Initializes and starts the background frame grabber and detector threads."""
    logger.info("Attempting to start background threads...") # Use info
    stop_event.clear() # Reset stop event

    # Clear frame queue
    while not frame_queue.empty():
        try: frame_queue.get_nowait()
        except queue.Empty: break

    # Reset shared data structures
    with config.frame_lock:
        config.latest_frame = None
    with config.detections_lock:
        config.latest_detections = {"results": None, "frame_shape": None}
    # Clear tracking data as well
    config.track_history.clear()
    config.tracked_objects_info.clear()

    # Initialize FrameGrabber
    capture_grabber = FrameGrabber(
        stream_url="rtsp://admin:QxT638_!1@192.168.0.55:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif",
        frame_queue=frame_queue, # Pass the queue for detections
        stop_event=stop_event
    )
    # Initialize ObjectDetector (doesn't need queues in constructor now)
    detection_worker = ObjectDetector()

    # Start threads
    capture_grabber.start()
    time.sleep(0.5) # Give grabber a moment to start
    detection_worker.start()

    # Check if threads started successfully
    if capture_grabber.thread.is_alive() and detection_worker.thread.is_alive():
        logger.info("Background threads started successfully.") # Use info
    else:
        logger.error("Error: One or more background threads failed to start.") # Use error
        # Attempt cleanup if threads didn't start
        if capture_grabber and capture_grabber.thread.is_alive(): capture_grabber.stop()
        if detection_worker and detection_worker.thread.is_alive(): detection_worker.stop()


def stop_threads():
    """Signals background threads to stop and waits for them to join."""
    logger.info("Signalling threads to stop...") # Use info
    stop_event.set() # Signal threads to stop their loops

    # Wait a brief moment for threads to acknowledge the event
    time.sleep(0.2)

    # Stop and join threads
    if capture_grabber:
        logger.info("Stopping frame grabber...") # Use info
        capture_grabber.stop() # Calls join inside
    if detection_worker:
        logger.info("Stopping object detector...") # Use info
        detection_worker.stop() # Calls join inside

    # Reset global references
    capture_grabber = None
    detection_worker = None
    logger.info("Background threads stop sequence completed.") # Use info


if __name__ == '__main__':
    try:
        # Start capture and detection threads for decoupled streaming
        start_threads()
        logger.info("Starting Flask development server...") # Use info
        # Disable Flask's default logger if using basicConfig, or configure Flask's logger
        # app.logger.disabled = True # Option 1: Disable Flask default logger
        log = logging.getLogger('werkzeug') # Option 2: Silence Werkzeug logger
        log.setLevel(logging.WARNING)

        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.") # Use info
    finally:
        logger.info("Shutting down threads...") # Use info
        stop_threads()
        logger.info("Server shut down.") # Use info

