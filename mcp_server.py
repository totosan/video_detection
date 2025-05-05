import logging
import threading
import requests # Import requests library
import json # Import json for parsing
from mcp.server.fastmcp import FastMCP
# Remove direct import of DetectionSystem
# from detection_system import DetectionSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Remove Global Detection System Instance and Locks ---
# detection_system_instance = None
# detection_system_lock = threading.Lock()
# start_stop_lock = threading.Lock() # Lock specifically for start/stop operations

# --- Flask API Base URL ---
FLASK_API_BASE_URL = "http://localhost:3000/api" # Assuming Flask runs on port 3000

# --- MCP Server Setup ---
mcp = FastMCP("VideoDetectionServer")

# --- Remove get_detection_system function ---
# def get_detection_system():
#    ...

# --- Helper function for API calls ---
def make_api_request(endpoint: str, method: str = 'GET', **kwargs) -> dict:
    """Helper function to make requests to the Flask API."""
    url = f"{FLASK_API_BASE_URL}/{endpoint}"
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=5, **kwargs) # Add timeout
        elif method.upper() == 'POST':
            response = requests.post(url, timeout=5, **kwargs) # Add timeout
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return {"error": f"Unsupported HTTP method: {method}"}

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json() # Parse JSON response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {method} {url}: {e}")
        return {"error": f"API request failed: {e}"}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {method} {url}: {e}")
        return {"error": f"Invalid JSON response from API: {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error during API request to {method} {url}")
        return {"error": f"Unexpected error during API request: {e}"}

# --- MCP Tools (Modified) ---

# --- Remove start/stop tools ---
# @mcp.tool()
# def start_detection_system() -> str:
#    ...
#
# @mcp.tool()
# def stop_detection_system() -> str:
#    ...

@mcp.tool()
def get_system_status() -> dict:
    """
    Checks the running status of the detection system via the Flask API.

    Returns:
        dict: A dictionary indicating the running status.
              Example success: {"is_running": True}
              Example error: {"is_running": False, "error": "API request failed..."}
    """
    logger.info("MCP Tool: get_system_status called.")
    result = make_api_request("status")
    # Ensure the key matches the API response
    if "error" not in result and "running" in result:
        return {"is_running": result["running"]}
    elif "error" in result:
        return {"is_running": False, "error": result["error"]}
    else:
        logger.error("MCP Tool: get_system_status received unexpected API response.")
        return {"is_running": False, "error": "Unexpected API response format"}


@mcp.tool()
def get_current_detections() -> dict:
    """
    Retrieves the latest raw detection results (bounding boxes, confidences, classes) from the Flask API.
    This provides the immediate output of the object detector before tracking is applied.

    Returns:
        dict: A dictionary containing a list of current detections or an error message.
              Example success: {"detections": [{"box": [10, 20, 50, 60], "confidence": 0.95, "class_id": 0, "class_name": "person"}, ...]}
              Example error: {"error": "API request failed..."}
    """
    logger.info("MCP Tool: get_current_detections called.")
    result = make_api_request("current_detections")
    if "error" in result:
        logger.error(f"MCP Tool: Error from API for get_current_detections: {result['error']}")
    # Assuming the API returns a dict like {"detections": [...]}, pass it through.
    # If the API returns just the list, wrap it: return {"detections": result}
    return {"detections": result['detections']} # Return the raw API response (or error dict)

@mcp.tool()
def get_tracked_objects() -> dict:
    """
    Retrieves information about objects currently being tracked by the system via the Flask API.
    This includes assigned track IDs and their current estimated positions.

    Returns:
        dict: A dictionary containing a list of tracked objects or an error message.
              Example success: {"tracked_objects": [{"track_id": 1, "box": [12, 22, 50, 60], "class_name": "person"}, {"track_id": 3, ...}, ...]}
              Example error: {"tracked_objects": [], "error": "API request failed..."}
    """
    logger.info("MCP Tool: get_tracked_objects called.")
    result = make_api_request("tracked_objects")
    if "error" in result:
        logger.error(f"MCP Tool: Error from API for get_tracked_objects: {result['error']}")
        # Return in the expected format for the original tool if needed, or just the error
        return {"tracked_objects": [], "error": result["error"]}
    else:
        # The API returns a list directly, wrap it in the expected dict format.
        return {"tracked_objects": result}


@mcp.tool()
def get_object_track_history() -> dict:
    """
    Retrieves the recent historical positions for currently or recently tracked objects via the Flask API.
    Useful for understanding object movement over the last few frames/seconds.

    Returns:
        dict: A dictionary where keys are track IDs and values are lists of recent bounding boxes for that track, or an error message.
              Example success: {"track_history": {1: [[10, 20, 50, 60], [11, 21, 50, 60]], 3: [[...], ...], ...}}
              Example error: {"track_history": {}, "error": "API request failed..."}
    """
    logger.info("MCP Tool: get_object_track_history called.")
    result = make_api_request("track_history")
    if "error" in result:
        logger.error(f"MCP Tool: Error from API for get_object_track_history: {result['error']}")
        return {"track_history": {}, "error": result["error"]}
    else:
        # Assuming the API returns the history dict directly, wrap it.
        return {"track_history": result}

# --- New Tools for Backend Annotation Control ---

@mcp.tool()
def toggle_backend_annotation() -> dict:
    """
    Toggles the generation of annotated video frames on the backend server via the Flask API.

    Returns:
        dict: A dictionary indicating the new status or an error message.
              Example success: {"backend_annotation_enabled": True}
              Example error: {"error": "API request failed..."}
    """
    logger.info("MCP Tool: toggle_backend_annotation called.")
    result = make_api_request("backend_annotation/toggle", method='POST')
    if "error" in result:
        logger.error(f"MCP Tool: Error from API for toggle_backend_annotation: {result['error']}")
    # Return the raw API response (or error dict)
    return result

@mcp.tool()
def get_backend_annotation_status() -> dict:
    """
    Gets the current status (enabled/disabled) of backend annotation generation via the Flask API.

    Returns:
        dict: A dictionary indicating the current status or an error message.
              Example success: {"backend_annotation_enabled": False}
              Example error: {"error": "API request failed..."}
    """
    logger.info("MCP Tool: get_backend_annotation_status called.")
    result = make_api_request("backend_annotation/status", method='GET')
    if "error" in result:
        logger.error(f"MCP Tool: Error from API for get_backend_annotation_status: {result['error']}")
    # Return the raw API response (or error dict)
    return result

# --- End New Tools ---

# --- New Tools for Snapshots ---

@mcp.tool()
def get_snapshot() -> dict:
    """
    Requests a snapshot of the latest processed frame from the Flask API.

    Returns:
        dict: A dictionary indicating success or failure, and the URL to fetch the snapshot.
              Example success: {"status": "success", "url": "http://localhost:3000/snapshot"}
              Example error: {"status": "error", "message": "API request failed...", "url": "http://localhost:3000/snapshot"}
    """
    logger.info("MCP Tool: get_snapshot called.")
    # Correct URL construction (assuming Flask runs at root)
    flask_app_base_url = FLASK_API_BASE_URL.replace("/api", "") # Derive base URL
    url = f"{flask_app_base_url}/snapshot"
    try:
        # Use requests directly as this endpoint returns an image, not JSON
        response = requests.get(url, timeout=5, stream=True) # stream=True might be useful if checking headers first
        response.raise_for_status() # Check for HTTP errors
        # Don't try to parse JSON, just confirm success
        logger.info(f"MCP Tool: Snapshot request to {url} successful.")
        return {"status": "success", "url": url}
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Tool: API request failed for get_snapshot at {url}: {e}")
        return {"status": "error", "message": f"API request failed: {e}", "url": url}
    except Exception as e:
        logger.exception(f"MCP Tool: Unexpected error during get_snapshot request to {url}")
        return {"status": "error", "message": f"Unexpected error: {e}", "url": url}

@mcp.tool()
def get_raw_snapshot() -> dict:
    """
    Requests a raw snapshot directly from the RTSP stream via the Flask API.

    Returns:
        dict: A dictionary indicating success or failure, and the URL to fetch the raw snapshot.
              Example success: {"status": "success", "url": "http://localhost:3000/raw_snapshot"}
              Example error: {"status": "error", "message": "API request failed...", "url": "http://localhost:3000/raw_snapshot"}
    """
    logger.info("MCP Tool: get_raw_snapshot called.")
    # Correct URL construction (assuming Flask runs at root)
    flask_app_base_url = FLASK_API_BASE_URL.replace("/api", "") # Derive base URL
    url = f"{flask_app_base_url}/raw_snapshot"
    try:
        # Use requests directly as this endpoint returns an image, not JSON
        response = requests.get(url, timeout=10, stream=True) # Longer timeout for direct capture
        response.raise_for_status() # Check for HTTP errors
        # Don't try to parse JSON, just confirm success
        logger.info(f"MCP Tool: Raw snapshot request to {url} successful.")
        return {"status": "success", "url": url}
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP Tool: API request failed for get_raw_snapshot at {url}: {e}")
        return {"status": "error", "message": f"API request failed: {e}", "url": url}
    except Exception as e:
        logger.exception(f"MCP Tool: Unexpected error during get_raw_snapshot request to {url}")
        return {"status": "error", "message": f"Unexpected error: {e}", "url": url}

# --- End New Tools ---


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# --- Main Execution (Simplified) ---
if __name__ == "__main__":
    logger.info("Starting MCP Server (API Client Mode)...")
    try:
        # No need to initialize or manage DetectionSystem here
        logger.info("Running MCP server...")
        mcp.run(transport='stdio')
    except Exception as e:
        logger.exception("An unexpected error occurred while running the MCP server.")
    finally:
        # No system shutdown needed here, Flask app manages it
        logger.info("MCP Server finished.")

