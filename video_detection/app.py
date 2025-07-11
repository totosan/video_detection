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
import io # Added for image byte handling
import zmq # <--- ADDED IMPORT
import json # <--- ADDED IMPORT (was missing in previous thought, but present in my last code generation for app.py)

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

# --- ZeroMQ Server Setup ---
ZMQ_CONTEXT = None
ZMQ_IMAGE_PULL_SOCKET = None
ZMQ_RESULTS_REP_SOCKET = None
ZMQ_SERVER_THREAD = None
ZMQ_STOP_EVENT = threading.Event()

def zmq_detection_server_thread_func():
    """
    Thread function to run the ZeroMQ server for image detection.
    Listens for images on a PULL socket and sends results on a REP socket.
    """
    global ZMQ_CONTEXT, ZMQ_IMAGE_PULL_SOCKET, ZMQ_RESULTS_REP_SOCKET, detection_system

    # Changed to TCP, ensure these match the client (vision_node.py) and Docker port mappings if used
    image_receiver_endpoint = os.environ.get("DETECTION_IMAGE_ENDPOINT", "tcp://*:5555")
    results_sender_endpoint = os.environ.get("DETECTION_RESULTS_ENDPOINT", "tcp://*:5556")

    try:
        ZMQ_CONTEXT = zmq.Context()
        
        ZMQ_IMAGE_PULL_SOCKET = ZMQ_CONTEXT.socket(zmq.PULL)
        ZMQ_IMAGE_PULL_SOCKET.setsockopt(zmq.RCVTIMEO, 1000) # Timeout for recv to check stop event
        ZMQ_IMAGE_PULL_SOCKET.bind(image_receiver_endpoint)
        logger.info(f"ZeroMQ: Image PULL socket bound to {image_receiver_endpoint}")

        ZMQ_RESULTS_REP_SOCKET = ZMQ_CONTEXT.socket(zmq.REP)
        ZMQ_RESULTS_REP_SOCKET.setsockopt(zmq.RCVTIMEO, 1000) # Timeout for recv to check stop event
        ZMQ_RESULTS_REP_SOCKET.bind(results_sender_endpoint)
        logger.info(f"ZeroMQ: Results REP socket bound to {results_sender_endpoint}")

        logger.info("ZeroMQ detection server thread started. Waiting for images...")

        poller = zmq.Poller()
        poller.register(ZMQ_IMAGE_PULL_SOCKET, zmq.POLLIN)
        poller.register(ZMQ_RESULTS_REP_SOCKET, zmq.POLLIN)

        while not ZMQ_STOP_EVENT.is_set():
            try:
                logger.debug("ZMQ server: Polling sockets...")
                socks = dict(poller.poll(timeout=1000)) # Poll with a timeout to check stop_event

                if ZMQ_IMAGE_PULL_SOCKET in socks and socks[ZMQ_IMAGE_PULL_SOCKET] == zmq.POLLIN:
                    logger.debug("ZMQ server: Image socket has data. Attempting to receive image bytes...")
                    img_bytes = ZMQ_IMAGE_PULL_SOCKET.recv(flags=zmq.NOBLOCK) # Use NOBLOCK as poller indicated readability
                    logger.info(f"ZMQ server: Received {len(img_bytes)} image bytes.")

                    # Now wait for the "detect" signal on the REP socket
                    logger.debug("ZMQ server: Waiting for 'detect' signal on REP socket...")
                    
                    # We need to poll specifically for the results_socket now
                    # This inner poll is tricky because REP expects a strict recv/send sequence.
                    # If we recv image, we MUST wait for a recv on REP then send on REP.
                    
                    # Let's simplify: assume "detect" signal comes quickly after image.
                    # The REP socket should also have a timeout.
                    try:
                        logger.debug("ZMQ server: Attempting to receive 'detect' signal...")
                        signal = ZMQ_RESULTS_REP_SOCKET.recv_string() # This will use RCVTIMEO if no message
                        logger.info(f"ZMQ server: Received signal: '{signal}'")

                        if signal == "detect":
                            logger.debug("ZMQ server: Decoding image...")
                            cv_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if cv_image is None:
                                logger.error("ZMQ server: Failed to decode image.")
                                ZMQ_RESULTS_REP_SOCKET.send_json({"error": "Failed to decode image"})
                                continue

                            logger.debug("ZMQ server: Processing image with detection system...")
                            detections = detection_system.process_single_image(cv_image)
                            logger.info(f"ZMQ server: Detection complete. Found {len(detections)} objects. Sending results.")
                            ZMQ_RESULTS_REP_SOCKET.send_json({"detections": detections})
                            logger.debug("ZMQ server: Results sent.")
                        else:
                            logger.warn(f"ZMQ server: Received unknown signal '{signal}'. Sending error.")
                            ZMQ_RESULTS_REP_SOCKET.send_json({"error": f"Unknown signal: {signal}"})
                    
                    except zmq.error.Again:
                        logger.warn("ZMQ server: Timeout waiting for 'detect' signal on REP socket after receiving image. No reply sent.")
                        # This is problematic for REP socket state. It might need a reset or a dummy send if protocol allows.
                        # For now, we just log. The client will time out.
                        # To prevent REP socket from getting stuck, we might need to send an error response here.
                        # However, a REP socket *must* reply if it has received a request.
                        # If recv_string timed out, it means no request was fully received on REP,
                        # so we should NOT send. The issue is client-side if PUSH was sent but REQ was not.
                        # But the client logs show REQ was sent.
                        # This points to a fundamental ordering issue or the REP socket not seeing the REQ.
                        # Let's ensure the client isn't sending REQ *before* PUSH is fully processed.
                        # The current client logic is PUSH then REQ.
                        # The server logic is PULL then REP. This should match.

                # Check if stop_event was set during poll or processing
                if ZMQ_STOP_EVENT.is_set():
                    logger.info("ZMQ server: Stop event detected, breaking loop.")
                    break
            
            except zmq.error.Again:
                # This will catch timeouts from poller.poll() if image_socket.recv() was not called
                # or if RCVTIMEO on sockets themselves trigger if not using NOBLOCK with poller.
                logger.debug("ZMQ server: Poll timed out, no messages received. Checking stop event.")
                if ZMQ_STOP_EVENT.is_set():
                    logger.info("ZMQ server: Stop event detected after poll timeout, breaking loop.")
                    break
                continue # Continue to next poll iteration
                
            except Exception as e:
                logger.error(f"ZMQ server: Error in detection server loop: {e}", exc_info=True)
                # If it's a REP socket error, it might be stuck.
                # A simple break/continue might not be enough.
                # For now, just log and continue, hoping client timeout/retry handles it.
                if isinstance(e, zmq.error.ZMQError) and ZMQ_RESULTS_REP_SOCKET and not ZMQ_RESULTS_REP_SOCKET.closed:
                    try:
                        # Try to send an error if we are in a state where a send is expected
                        # This is very hard to get right without knowing the exact state.
                        # ZMQ_RESULTS_REP_SOCKET.send_json({"error": "Server loop exception"})
                        logger.error("ZMQ server: A ZMQError occurred. The REP socket might be in an inconsistent state.")
                    except Exception as send_e:
                        logger.error(f"ZMQ server: Error trying to send error response: {send_e}")
                if ZMQ_STOP_EVENT.is_set():
                    break
                time.sleep(0.1) # Avoid tight loop on persistent error

        logger.info("ZMQ detection server thread stopping.")
    except Exception as e:
        logger.exception("Fatal error in ZeroMQ detection server thread setup")
    finally:
        if ZMQ_IMAGE_PULL_SOCKET:
            ZMQ_IMAGE_PULL_SOCKET.close()
        if ZMQ_RESULTS_REP_SOCKET:
            ZMQ_RESULTS_REP_SOCKET.close()
        # Context termination is handled in cleanup_on_exit
        logger.info("ZeroMQ detection server thread stopped.")

# --- End ZeroMQ Server Setup ---

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
    """API endpoint to get the latest raw detection results for client-side drawing.
       This version is 'light' because it does not attempt to add cropped images.
       It directly returns the serializable detections from the detection system.
       Filtering is applied here based on the currently set filters.
    """
    try:
        data_from_system = detection_system.get_current_detections_data()
        
        detections_for_api = data_from_system.get('detections', [])
        frame_shape_for_api = data_from_system.get('frame_shape', None)

        # --- Apply Filters ---
        track_id_filter = detection_system.get_track_id_filter()
        label_filter = detection_system.get_label_filter()

        if track_id_filter is not None:
            detections_for_api = [d for d in detections_for_api if d.get('track_id') == track_id_filter]
        
        if label_filter: # If the list is not empty
            detections_for_api = [d for d in detections_for_api if d.get('label') in label_filter]
        # --- End of Filters ---

        response_data = {
            'detections': detections_for_api,
            'frame_shape': frame_shape_for_api
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.exception("API: Error getting or serializing current_detections_light data")
        return jsonify({"error": "Failed to get current detections data", "detections": [], "frame_shape": None}), 500

@app.route('/api/current_detections')
def api_current_detections():
    """API endpoint to get the latest raw detection results for client-side drawing,
       including cropped images for each detection.
       Filtering is applied here based on the currently set filters.
    """
    try:
        data_from_system = detection_system.get_current_detections_data()
        
        detections_for_api = data_from_system.get('detections', [])
        frame_shape_for_api = data_from_system.get('frame_shape', None)

        # --- Apply Filters ---
        track_id_filter = detection_system.get_track_id_filter()
        label_filter = detection_system.get_label_filter()

        if track_id_filter is not None:
            detections_for_api = [d for d in detections_for_api if d.get('track_id') == track_id_filter]

        if label_filter: # If the list is not empty
            detections_for_api = [d for d in detections_for_api if d.get('label') in label_filter]
        # --- End of Filters ---

        # Add detection images (cropped regions) for each detection
        latest_frame_for_cropping = detection_system.get_latest_frame()

        if latest_frame_for_cropping is not None:
            for detection in detections_for_api: # Iterate over the list of detection dicts
                try:
                    box = detection.get('box')  # [x_min, y_min, x_max, y_max]
                    if box and len(box) == 4:
                        x_min, y_min, x_max, y_max = map(int, box)
                        # Ensure coordinates are valid before cropping
                        h, w = latest_frame_for_cropping.shape[:2]
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(w, x_max), min(h, y_max)
                        
                        if x_max > x_min and y_max > y_min:
                            cropped_image = latest_frame_for_cropping[y_min:y_max, x_min:x_max]
                            if cropped_image.size > 0: # Check if cropped image is not empty
                                ret, buffer = cv2.imencode('.jpg', cropped_image)
                                if ret:
                                    detection['image'] = base64.b64encode(buffer).decode('utf-8')
                                else:
                                    logger.warning("current_detections: Could not encode cropped detection image to JPEG")
                            else:
                                logger.warning(f"current_detections: Cropped image is empty for box {box}")
                        else:
                             logger.warning(f"current_detections: Invalid box coordinates for cropping: {box}")
                    else:
                        logger.warning("current_detections: Invalid bounding box format for image cropping")
                except Exception as e:
                    logger.exception("current_detections: Error processing detection image")
        else:
            logger.warning("current_detections: No latest frame available for cropping detection images")

        response_data = {
            'detections': detections_for_api,
            'frame_shape': frame_shape_for_api
        }
        return jsonify(response_data)
    except Exception as e:
        logger.exception("API: Error getting or serializing current_detections data with images")
        return jsonify({"error": "Failed to get current detections data with images", "detections": [], "frame_shape": None}), 500
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
    """Returns a single JPEG snapshot, preferring the annotated frame, then raw frame."""
    if not detection_system.is_running():
        logger.warning("snapshot: Detection system not running.")
        return ("Detection system not running", 503)

    logger.debug("Snapshot: Retrieving frame from detection_system.")
    frame_to_send = None
    
    # Try to get the annotated frame first
    annotated_frame = detection_system.get_latest_annotated_frame()
    if annotated_frame is not None:
        logger.debug("Snapshot: Using latest annotated frame.")
        frame_to_send = annotated_frame
    else:
        logger.debug("Snapshot: Annotated frame not available, trying latest raw frame.")
        raw_frame = detection_system.get_latest_frame()
        if raw_frame is not None:
            logger.debug("Snapshot: Using latest raw frame.")
            frame_to_send = raw_frame

    if frame_to_send is None:
        logger.info("Snapshot: No frame available (neither annotated nor raw).")
        return ("No frame available", 503)

    ret, buffer = cv2.imencode('.jpg', frame_to_send)
    if not ret:
        logger.error("Snapshot: Error encoding frame.")
        return ("Error encoding frame", 500)
    
    logger.debug("Snapshot: Returning JPEG image.")
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/raw_snapshot')
def raw_snapshot():
    """Returns a single JPEG snapshot directly from the currently selected source (no threads)."""
    logger.debug("Raw snapshot: Attempting direct capture from current detection system source.")
    
    cap = None
    # Use the current source from detection_system, not the static config
    source_to_open = str(getattr(detection_system, 'source_identifier', RTSP_STREAM_URL))
    source_type = getattr(detection_system, 'source_type', None)
    is_device = False

    if source_type == 'device' or (source_to_open.isdigit() and source_type is None):
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
            logger.error("Raw snapshot: Video source string is empty.")
            return ("Video source string is empty", 503)
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

@app.route('/api/set_track_id_filter', methods=['POST'])
def set_track_id_filter():
    """API endpoint to set the track ID filter."""
    try:
        data = request.get_json()
        if not data or 'track_id' not in data:
            logger.warning("API: Invalid request to set track ID filter. 'track_id' missing.")
            return jsonify({"error": "Missing 'track_id' in request"}), 400

        track_id = data['track_id']
        detection_system.set_track_id_filter(track_id)
        logger.info(f"API: Track ID filter set to {track_id}.")
        return jsonify({"message": "Track ID filter set successfully", "track_id_filter": track_id}), 200
    except Exception as e:
        logger.exception("API: Error setting track ID filter")
        return jsonify({"error": "Failed to set track ID filter", "details": str(e)}), 500

@app.route('/api/get_track_id_filter', methods=['GET'])
def get_track_id_filter():
    """API endpoint to get the current track ID filter."""
    try:
        track_id_filter = detection_system.get_track_id_filter()
        logger.info(f"API: Current Track ID filter: {track_id_filter}")
        return jsonify({"track_id_filter": track_id_filter}), 200
    except Exception as e:
        logger.exception("API: Error retrieving track ID filter")
        return jsonify({"error": "Failed to retrieve track ID filter", "details": str(e)}), 500

@app.route('/api/set_object_filter', methods=['POST'])
def set_object_filter():
    """API endpoint to set the object filter for displaying specific labels."""
    try:
        # Expecting {'object_filter': ['person', 'car']} or {'object_filter': []}
        filter_data = request.json.get('object_filter', None)
        
        # detection_system.set_object_filter handles None and type checking
        detection_system.set_object_filter(filter_data)

        status_message = f"Filter set to labels: {filter_data}" if filter_data else "Label filter cleared"
        logger.info(f"API: {status_message}")
        return jsonify({"message": status_message, "object_filter": filter_data}), 200

    except Exception as e:
        logger.exception("API: Error setting object filter")
        return jsonify({"error": "Failed to set object filter", "details": str(e)}), 500

@app.route('/api/get_object_filter', methods=['GET'])
def get_object_filter():
    """API endpoint to get the current object filter."""
    try:
        current_filter = detection_system.get_label_filter()
        logger.debug(f"API: Getting object filter: {current_filter}")
        return jsonify({"object_filter": current_filter}), 200
    except Exception as e:
        logger.exception("API: Error getting object filter")
        return jsonify({"error": "Failed to get object filter", "object_filter": []}), 500

# --- API Endpoint for Single Image Detection ---
@app.route('/api/detect', methods=['POST'])
def api_detect_objects():
    """
    API endpoint to detect objects in an uploaded image.
    Expects a POST request with 'multipart/form-data' encoding.
    The image file should be sent under the form field name 'image'.
    Supported image formats are those decodable by OpenCV (e.g., JPEG, PNG).
    Returns a JSON response with a list of detected objects.
    Each object in the list is a dictionary, e.g.:
    {'box': [x_min, y_min, x_max, y_max], 'label': 'person', 'confidence': 0.9}
    """
    if 'image' not in request.files:
        logger.warning("API /api/detect: No image file in request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("API /api/detect: No selected file.")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image file into a numpy array
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        cv_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if cv_image is None:
            logger.error("API /api/detect: Could not decode image.")
            return jsonify({"error": "Could not decode image"}), 400

        # Process the image using the detection system
        request_time = time.time() # For potential timing analysis
        detections, _ = detection_system.process_single_image(cv_image, client_request_time=request_time)
        
        logger.info(f"API /api/detect: Processed {file.filename}, found {len(detections)} detections.")
        return jsonify({"detections": detections}) # Ensure consistent response format

    except Exception as e:
        logger.exception("API /api/detect: Error processing image")
        return jsonify({"error": str(e)}), 500
# ---------------------------------------------

# --- API Endpoints ---
@app.route('/api/select_closest_object', methods=['POST'])
def select_closest_object():
    """
    Selects the object closest to the given (x, y) coordinates and sets the track ID filter.
    """
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({"error": "Missing x or y coordinates"}), 400

    # The frontend sends normalized coordinates, but the backend expects pixel coordinates.
    # We need the frame dimensions to convert them back.
    # Let's get the last processed frame dimensions from the detection system.
    frame_height, frame_width = detection_system.get_last_frame_dimensions()

    if frame_width == 0 or frame_height == 0:
        return jsonify({"error": "Backend not ready, frame dimensions unknown."}), 503

    # Convert normalized coordinates to pixel coordinates
    pixel_x = int(x * frame_width)
    pixel_y = int(y * frame_height)

    logger.info(f"Received click at normalized ({x:.2f}, {y:.2f}), pixel ({pixel_x}, {pixel_y})")

    # Get the current detections (unfiltered)
    all_detections = detection_system.get_current_detections()

    closest_object = None
    min_distance = float('inf')

    for det in all_detections:
        track_id = det.get('track_id')
        box = det.get('box') # (x1, y1, x2, y2)
        mask = det.get('mask') # Optional mask

        if not track_id or not box:
            continue

        # Determine the center of the object
        if mask is not None and len(mask) > 0:
            # Calculate centroid of the mask
            M = cv2.moments(np.array(mask, dtype=np.int32))
            if M["m00"] > 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                # Fallback for zero-area contour
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
        else:
            # Use center of the bounding box
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

        # Calculate Euclidean distance from click to object center
        distance = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)

        if distance < min_distance:
            min_distance = distance
            closest_object = det

    if closest_object:
        selected_track_id = closest_object.get('track_id')
        logger.info(f"Closest object found: track_id={selected_track_id} with distance {min_distance:.2f}")
        # Set the system's track ID filter
        detection_system.set_track_id_filter(selected_track_id)
        return jsonify({
            "success": True, 
            "message": f"Track ID filter set to {selected_track_id}",
            "selected_track_id": selected_track_id
        })
    else:
        logger.info("No objects detected, cannot select closest.")
        return jsonify({"error": "No objects found to select from"}), 404

@app.route('/api/current_detections')
def get_current_detections():
    """Returns the current list of detected objects, optionally filtered."""
    try:
        # Get query parameters for filtering
        track_id_filter = request.args.get('track_id', type=int)
        label_filter = request.args.getlist('label')

        detections = detection_system.get_current_detections()

        # Apply track ID filter if provided
        if track_id_filter is not None:
            detections = [d for d in detections if d.get('track_id') == track_id_filter]

        # Apply label filter if provided
        if label_filter:
            detections = [d for d in detections if d.get('label') in label_filter]

        # Serialize detections for JSON response
        serialized_detections = []
        for det in detections:
            # Basic serialization
            serialized_det = {
                'track_id': det.get('track_id'),
                'label': det.get('label'),
                'confidence': det.get('confidence'),
                'box': det.get('box'), # Assuming box is already a serializable format
                # Add more fields as needed
            }

            # If you have complex types, convert them here
            # For example, if 'box' is a numpy array, convert to list: 'box': det['box'].tolist()

            serialized_detections.append(serialized_det)

        return jsonify(serialized_detections), 200
    except Exception as e:
        logger.exception("API: Error getting current detections")
        return jsonify({"error": "Failed to get current detections"}), 500
# ------------------------------------

# --- Graceful Shutdown --- 
def cleanup_on_exit():
    global ZMQ_STOP_EVENT, ZMQ_SERVER_THREAD, ZMQ_CONTEXT
    logger.info("Flask app exiting...")
    
    # Signal ZMQ thread to stop and wait for it
    if ZMQ_SERVER_THREAD and ZMQ_SERVER_THREAD.is_alive():
        logger.info("Stopping ZeroMQ server thread...")
        ZMQ_STOP_EVENT.set()
        ZMQ_SERVER_THREAD.join(timeout=5.0) # Wait for thread to finish
        if ZMQ_SERVER_THREAD.is_alive():
            logger.warning("ZeroMQ server thread did not stop in time.")
    
    if ZMQ_CONTEXT:
        logger.info("Terminating ZeroMQ context...")
        ZMQ_CONTEXT.term() # Terminate context after sockets are closed by the thread
        logger.info("ZeroMQ context terminated.")

    logger.info("Stopping detection system...")
    detection_system.stop()
    logger.info("Detection system stopped.")
    logger.info("Cleanup complete.")


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

        # --- Start ZeroMQ Server Thread ---
        logger.info("Starting ZeroMQ detection server thread...")
        ZMQ_STOP_EVENT.clear() # Ensure event is clear before starting
        ZMQ_SERVER_THREAD = threading.Thread(target=zmq_detection_server_thread_func, daemon=True)
        ZMQ_SERVER_THREAD.start()
        # ----------------------------------

        logger.info("Starting Flask development server...") # Use info
        # Disable Flask's default logger if using basicConfig, or configure Flask's logger
        log = logging.getLogger('werkzeug') # Silence Werkzeug logger
        log.setLevel(logging.WARNING)
        
        # Get host and port from environment variables or use defaults
        host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
        port = int(os.environ.get('FLASK_RUN_PORT', 3000))
        
        app.run(host=host, port=port, debug=False, use_reloader=False) # use_reloader=False is important for threads


    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Initiating shutdown...")
    except Exception as e:
        logger.exception("Failed to start Flask application")
    finally:
        # cleanup_on_exit will be called by atexit
        logger.info("Application shutdown sequence initiated or completed.")


