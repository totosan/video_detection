import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import time
import threading # Keep threading if needed for Flask or other parts
import numpy as np # Keep numpy if needed elsewhere
from flask import Flask, render_template, Response, jsonify, request
import logging # Import logging
import atexit # To ensure cleanup on exit

# Import static config and the new system manager
from config import STATIC_FOLDER, TEMPLATE_FOLDER, RTSP_STREAM_URL # Only import static config
from detection_system import DetectionSystem

# Initialize Flask app
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Get logger
logger = logging.getLogger(__name__) # Use module name for logger

# --- Instantiate the Detection System ---
try:
    detection_system = DetectionSystem()
except Exception as e:
    logger.exception("Failed to initialize DetectionSystem. Flask app cannot start.")
    # Exit or handle appropriately if the core system fails to init
    sys.exit(1)
# ---------------------------------------

def generate_frames(lock, frame_source_func):
    """Generator function to yield frames for streaming."""
    while True:
        time.sleep(0.03) # Limit frame rate slightly
        frame = frame_source_func() # Call the getter method passed
        if frame is None:
            # Optional: Send a placeholder image if no frame is available
            # logger.debug("generate_frames: No frame available, skipping yield.")
            continue

        try:
            # Lock is now managed within the getter in DetectionSystem, but keep it here
            # if direct access to a shared resource outside DetectionSystem is needed.
            # If getters handle locking, this lock might be redundant.
            # For now, assume getters are thread-safe and don't require external lock here.
            # with lock: # Re-evaluate if this lock is needed based on getter implementation
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("Could not encode frame to JPEG") # Use warning
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.exception(f"Error encoding or yielding frame: {e}") # Use exception
            # Consider breaking or handling differently if errors persist

@app.route('/')
def index():
    """Serves the main HTML page."""
    logger.debug("Serving index.html")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route for the raw camera feed."""
    logger.info("Raw video feed requested.") # Use info
    # Pass the getter method from detection_system
    return Response(generate_frames(detection_system.frame_lock, detection_system.get_latest_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_annotated')
def video_feed_annotated():
    """Video streaming route for the feed with detections."""
    logger.info("Annotated video feed requested.") # Use info
    # Pass the getter method from detection_system
    return Response(generate_frames(detection_system.annotated_frame_lock, detection_system.get_latest_annotated_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/tracked_objects')
def api_tracked_objects():
    """API endpoint to get the latest tracked object information."""
    current_time = time.time()
    objects_list = []
    # Get tracked info from the detection system
    tracked_info = detection_system.get_tracked_objects_info()

    # Create a copy of the keys to avoid runtime errors if the dict changes during iteration
    current_tracked_ids = list(tracked_info.keys())

    for track_id in current_tracked_ids:
        if track_id in tracked_info: # Check if ID still exists
            info = tracked_info[track_id]
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

    try:
        return jsonify(objects_list) # Return the list
    except Exception as e:
        logger.exception("API: Error converting tracked_objects list to JSON") # Use exception
        return jsonify({"error": "Failed to serialize tracked objects data"}), 500

@app.route('/snapshot')
def snapshot():
    """Returns a single JPEG snapshot from the latest captured frame."""
    # Ensure detection system is running (optional check)
    if not detection_system.is_running():
        logger.warning("snapshot: Detection system not running.")
        # Optionally try restarting it, or just return error
        # detection_system.start() # Be careful with restarting logic
        return ("Detection system not running", 503)

    logger.debug("Snapshot: Retrieving from detection_system.")
    frame = detection_system.get_latest_frame()

    if frame is None:
        logger.info("Snapshot: No frame available.") # Use info
        return ("No frame available", 503)

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        logger.error("Snapshot: Error encoding frame.") # Use error
        return ("Error encoding frame", 500)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/raw_snapshot')
def raw_snapshot():
    """Returns a single JPEG snapshot directly from the RTSP stream (no threads)."""
    logger.debug("Raw snapshot: Opening direct RTSP capture.")
    # Use the URL from config
    cap = cv2.VideoCapture(RTSP_STREAM_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Raw snapshot: Unable to open stream {RTSP_STREAM_URL}") # Use error
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

# --- Graceful Shutdown --- 
def cleanup_on_exit():
    logger.info("Flask app exiting, stopping detection system...")
    detection_system.stop()
    logger.info("Detection system stopped.")

atexit.register(cleanup_on_exit)
# -------------------------

if __name__ == '__main__':
    try:
        # Start the detection system's background threads
        detection_system.start()

        logger.info("Starting Flask development server...") # Use info
        # Disable Flask's default logger if using basicConfig, or configure Flask's logger
        log = logging.getLogger('werkzeug') # Silence Werkzeug logger
        log.setLevel(logging.WARNING)

        # Run Flask app (threaded=True is important for handling multiple requests)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.") # Use info
        # Cleanup is handled by atexit
    except Exception as e:
        logger.exception("An unexpected error occurred during Flask app execution.")
    finally:
        # Ensure cleanup runs even if atexit fails in some scenarios (though it should work)
        if detection_system.is_running():
             logger.warning("Cleanup: Detection system still seems to be running, attempting stop again.")
             cleanup_on_exit()
        logger.info("Server shut down process completed.")

