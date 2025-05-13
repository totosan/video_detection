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
import collections # Import collections for deque type checking
import base64 # Import base64 for image encoding
import signal

# Import static config and the new system manager
from config import STATIC_FOLDER, TEMPLATE_FOLDER, RTSP_STREAM_URL # Only import static config
from detection_system import DetectionSystem
from utilities.read_video_source import find_available_cameras # Added import

# Initialize Flask app
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Get logger
logger = logging.getLogger(__name__) # Use module name for logger

# Ensure all loggers respect the global logging level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.INFO)

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
    tracked_info = detection_system.get_tracked_objects_info()

    logger.debug(f"API called: Processing {len(tracked_info)} tracked objects")

    for track_id, info in tracked_info.items():
        time_since_seen = round(current_time - info.get('last_seen', current_time), 1)
        object_data = {
            'id': track_id,
            'name': info.get('name', 'Unknown'),
            'time_since_seen': time_since_seen
        }

        # Use image data directly from tracked_info if available
        if 'detection_image' in info and info['detection_image'] is not None:
            logger.debug(f"Track ID {track_id}: Found detection image in tracked_info, encoding to base64")
            try:
                ret, buffer = cv2.imencode('.jpg', info['detection_image'])
                if ret:
                    object_data['detection_image'] = base64.b64encode(buffer).decode('utf-8')
                else:
                    logger.warning(f"Track ID {track_id}: Failed to encode detection image from tracked_info")
            except Exception as e:
                logger.exception(f"Track ID {track_id}: Error encoding detection image from tracked_info: {e}")
        else:
            logger.debug(f"Track ID {track_id}: No detection image available in tracked_info")

        objects_list.append(object_data)

    logger.debug(f"API Response: {len(objects_list)} objects, {sum(1 for obj in objects_list if 'detection_image' in obj)} with images")
    return jsonify(objects_list)

# --- API Endpoint for Available Cameras ---
@app.route('/api/cams', methods=['GET'])
def api_cams():
    """API endpoint to get a list of available video sources."""
    try:
        cameras = find_available_cameras(max_cameras_to_check=5) # Limit check for speed
        logger.info(f"API: Found {len(cameras)} available cameras.")
        return jsonify(cameras)
    except Exception as e:
        logger.exception("API: Error finding available cameras")
        return jsonify({"error": "Failed to retrieve camera list", "details": str(e)}), 500
# -----------------------------------------

# --- API Endpoint for Setting Video Source ---
@app.route('/api/selected_videosource', methods=['POST'])
def set_selected_videosource():
    """API endpoint to set the video source for the detection system."""
    try:
        data = request.get_json()
        if not data or 'source_identifier' not in data:
            logger.warning("API: Invalid request to set video source. 'source_identifier' missing.")
            return jsonify({"error": "Missing 'source_identifier' in request"}), 400

        source_identifier = data['source_identifier']
        logger.info(f"API: Request to change video source to: {source_identifier}")

        # Assuming detection_system has a method to change its source
        # This method needs to be implemented in DetectionSystem class
        success, message = detection_system.change_video_source(source_identifier)

        if success:
            logger.info(f"API: Video source changed successfully to {source_identifier}.")
            return jsonify({"message": message, "new_source": source_identifier}), 200
        else:
            logger.error(f"API: Failed to change video source to {source_identifier}. Reason: {message}")
            return jsonify({"error": message, "requested_source": source_identifier}), 500

    except Exception as e:
        logger.exception("API: Error setting video source")
        return jsonify({"error": "Failed to set video source", "details": str(e)}), 500
# -------------------------------------------

# --- New API Endpoint for Current Detections ---
@app.route('/api/current_detections_light')
def api_current_detections_light():
    """API endpoint to get the latest raw detection results for client-side drawing."""
    try:
        data = detection_system.get_current_detections_data()
        # Ensure detections is always a list, even if None initially
        if data.get('detections') is None:
            data['detections'] = []

        return jsonify(data['detections'])  # Now contains JSON-serializable data and detection images
    except Exception as e:
        logger.exception("API: Error getting or serializing current detections data")
        return jsonify({"error": "Failed to get current detections data"}), 500

@app.route('/api/current_detections')
def api_current_detections():
    """API endpoint to get the latest raw detection results for client-side drawing."""
    try:
        data = detection_system.get_current_detections_data()
        # Ensure detections is always a list, even if None initially
        if data.get('detections') is None:
            data['detections'] = []

        # Add detection images (cropped regions) for each detection
        detection_image = detection_system.get_latest_frame()
        if detection_image is not None:
            for detection in data['detections']:
                try:
                    box = detection.get('box')  # [x_min, y_min, x_max, y_max]
                    if box and len(box) == 4:
                        x_min, y_min, x_max, y_max = map(int, box)
                        cropped_image = detection_image[y_min:y_max, x_min:x_max]
                        ret, buffer = cv2.imencode('.jpg', cropped_image)
                        if ret:
                            detection['image'] = base64.b64encode(buffer).decode('utf-8')
                        else:
                            logger.warning("current_detections: Could not encode cropped detection image to JPEG")
                    else:
                        logger.warning("current_detections: Invalid bounding box format")
                except Exception as e:
                    logger.exception("current_detections: Error processing detection image")
        else:
            logger.warning("current_detections: No latest frame available for cropping detection images")

        return jsonify(data)  # Now contains JSON-serializable data and detection images
    except Exception as e:
        logger.exception("API: Error getting or serializing current detections data")
        return jsonify({"error": "Failed to get current detections data"}), 500
# ---------------------------------------------

# --- API Endpoints for Backend Annotation Control ---
@app.route('/api/backend_annotation/toggle', methods=['POST'])
def toggle_backend_annotation():
    """Toggles the backend annotation generation on/off."""
    try:
        new_status = detection_system.toggle_backend_annotation()
        return jsonify({"backend_annotation_enabled": new_status})
    except Exception as e:
        logger.exception("API: Error toggling backend annotation")
        return jsonify({"error": "Failed to toggle backend annotation"}), 500

@app.route('/api/backend_annotation/status', methods=['GET'])
def get_backend_annotation_status():
    """Gets the current status of backend annotation generation."""
    try:
        status = detection_system.is_backend_annotation_enabled()
        return jsonify({"backend_annotation_enabled": status})
    except Exception as e:
        logger.exception("API: Error getting backend annotation status")
        return jsonify({"error": "Failed to get backend annotation status"}), 500
# --------------------------------------------------

# --- API Endpoint for System Status ---
@app.route('/api/status')
def api_status():
    """API endpoint to get the running status of the detection system."""
    try:
        status = detection_system.is_running()
        return jsonify({"running": status})
    except Exception as e:
        logger.exception("API: Error getting detection system status")
        return jsonify({"error": "Failed to get system status"}), 500
# ------------------------------------

# --- API Endpoint for Track History ---
@app.route('/api/track_history')
def api_track_history():
    """API endpoint to get the history of tracked objects."""
    try:
        # Get query parameter to determine if images should be included
        include_images = request.args.get('include_images', 'false').lower() == 'true'
        
        history = detection_system.get_track_history()
        # Convert history keys (track_id) to strings if they are not already,
        # as JSON keys must be strings.
        # Also, convert internal data structures if necessary for JSON serialization.
        serializable_history = {}
        for track_id, data in history.items():
             # Check if data is actually a dictionary before processing
             if isinstance(data, dict):
                 serializable_data = {}
                 for key, value in data.items():
                     if isinstance(value, collections.deque):
                         # Convert deque to list for JSON
                         serializable_data[key] = list(value)
                     # Add check for numpy arrays if they might be present
                     elif isinstance(value, np.ndarray):
                         serializable_data[key] = value.tolist() # Convert numpy array to list
                     # Add check for numpy numeric types
                     elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                             np.uint8, np.uint16, np.uint32, np.uint64,
                                             np.float_, np.float16, np.float32, np.float64)):
                         serializable_data[key] = float(value) if isinstance(value, (np.float_, np.float16, np.float32, np.float64)) else int(value)
                     else:
                         # Assume other types are directly serializable
                         serializable_data[key] = value
                 
                 # Add detection image if requested
                 if include_images:
                     try:
                         # Get image for this track_id from detection system
                         detection_image = detection_system.get_detection_image(track_id)
                         if detection_image is not None:
                             # Convert the image to JPEG bytes
                             ret, buffer = cv2.imencode('.jpg', detection_image)
                             if ret:
                                 # Convert to base64 for JSON
                                 jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                                 serializable_data['detection_image'] = jpg_as_text
                             else:
                                 logger.warning(f"Failed to encode detection image for track_id {track_id}")
                     except Exception as img_err:
                         logger.exception(f"Error getting detection image for track_id {track_id}: {img_err}")
                 
                 serializable_history[str(track_id)] = serializable_data
             else:
                 logger.warning(f"API: Skipping track_id {track_id} in history serialization because its data is not a dictionary (type: {type(data)}).")
                 # Optionally add placeholder: serializable_history[str(track_id)] = {"error": "Invalid data format"}

        return jsonify(serializable_history)
    except Exception as e:
        logger.exception("API: Error getting or serializing track history")
        return jsonify({"error": "Failed to get track history"}), 500
# ------------------------------------

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
    """Returns a single JPEG snapshot directly from the configured source (no threads)."""
    logger.debug("Raw snapshot: Attempting direct capture from configured source.")
    
    cap = None
    source_to_open = RTSP_STREAM_URL # From config.py
    is_device = False

    if source_to_open.isdigit():
        try:
            device_index = int(source_to_open)
            logger.debug(f"Raw snapshot: Source is device index {device_index}.")
            cap = cv2.VideoCapture(device_index)
            is_device = True
        except ValueError:
            logger.warning(f"Raw snapshot: Could not parse '{source_to_open}' as device index, trying as URL.")
            # Fall through to RTSP logic
            pass 
    
    if not is_device: # Either it wasn't a digit, or parsing as int failed
        if not source_to_open:
            logger.error("Raw snapshot: Video source URL is empty in config.")
            return ("Video source URL is empty", 503)
        logger.debug(f"Raw snapshot: Source is URL '{source_to_open}'. Attempting with FFMPEG backend.")
        # For RTSP or file paths, FFMPEG is generally a good choice
        cap = cv2.VideoCapture(source_to_open, cv2.CAP_FFMPEG)

    if cap is None or not cap.isOpened():
        logger.error(f"Raw snapshot: Unable to open source {source_to_open}")
        return ("Cannot open video source", 503)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        logger.error(f"Raw snapshot: Failed to grab frame from {source_to_open}.")
        return ("Failed to grab frame", 503)
    
    ret2, buf = cv2.imencode('.jpg', frame)
    if not ret2:
        logger.error(f"Raw snapshot: Error encoding frame from {source_to_open}.")
        return ("Error encoding frame", 500)
    
    logger.debug(f"Raw snapshot: Returning JPEG image from {source_to_open}.")
    return Response(buf.tobytes(), mimetype='image/jpeg')

# Add API endpoints in app.py
@app.route('/api/toggle_tracking', methods=['POST'])
def toggle_tracking():
    """API to toggle tracking and bounding box drawing."""
    try:
        new_status = detection_system.toggle_tracking_and_bounding_boxes()
        return jsonify({"tracking_enabled": new_status})
    except Exception as e:
        logger.exception("API: Error toggling tracking and bounding boxes")
        return jsonify({"error": "Failed to toggle tracking and bounding boxes"}), 500

@app.route('/api/tracking_status', methods=['GET'])
def tracking_status():
    """API to get the current status of tracking and bounding box drawing."""
    try:
        status = detection_system.is_tracking_and_bounding_boxes_enabled()
        return jsonify({"tracking_enabled": status})
    except Exception as e:
        logger.exception("API: Error getting tracking status")
        return jsonify({"error": "Failed to get tracking status"}), 500

@app.route('/api/set_object_filter', methods=['POST'])
def set_object_filter():
    """API to set the object filter for displaying specific labels."""
    try:
        print(f"Request data: {request.json}")  # Debugging line
        filter_data = request.json.get('object_filter', None)
        detection_system.set_object_filter(filter_data)
        return jsonify({"object_filter": filter_data})
    except Exception as e:
        logger.exception("API: Error setting object filter")
        return jsonify({"error": "Failed to set object filter"}), 500

@app.route('/api/get_object_filter', methods=['GET'])
def get_object_filter():
    """API to get the current object filter."""
    try:
        current_filter = detection_system.get_object_filter()
        return jsonify({"object_filter": current_filter})
    except Exception as e:
        logger.exception("API: Error getting object filter")
        return jsonify({"error": "Failed to get object filter"}), 500

# --- Graceful Shutdown --- 
def cleanup_on_exit():
    logger.info("Flask app exiting, stopping detection system...")
    detection_system.stop()
    logger.info("Detection system stopped.")

atexit.register(cleanup_on_exit)
# -------------------------

def handle_exit_signal(signum, frame):
    logger.info("Exit signal received. Stopping the application...")
    detection_system.stop()
    logger.info("Detection system stopped. Exiting application.")
    sys.exit(0)

# Register signal handlers for clean shutdown
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

if __name__ == '__main__':
    try:
        # Start the detection system's background threads
        detection_system.start()

        logger.info("Starting Flask development server...") # Use info
        # Disable Flask's default logger if using basicConfig, or configure Flask's logger
        log = logging.getLogger('werkzeug') # Silence Werkzeug logger
        log.setLevel(logging.WARNING)

        # Run Flask app (threaded=True is important for handling multiple requests)
        app.run(host='0.0.0.0', port=3000, debug=False, threaded=True, use_reloader=False)

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


