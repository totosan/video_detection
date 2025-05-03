\
import mcp
import logging
import threading
from detection_system import DetectionSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Detection System Instance ---
# Ensure only one instance is created and managed
detection_system_instance = None
detection_system_lock = threading.Lock()
start_stop_lock = threading.Lock() # Lock specifically for start/stop operations

def get_detection_system():
    """Gets or creates the singleton DetectionSystem instance."""
    global detection_system_instance
    with detection_system_lock:
        if detection_system_instance is None:
            logger.info("Creating DetectionSystem instance for MCP Server...")
            try:
                detection_system_instance = DetectionSystem()
                logger.info("DetectionSystem instance created successfully.")
            except Exception as e:
                logger.exception("Failed to initialize DetectionSystem.")
                # Propagate the error so MCP server doesn't start incorrectly
                raise RuntimeError("Could not initialize the core Detection System") from e
        return detection_system_instance

# --- MCP Server Setup ---
mcp_server = mcp.MCP(
    name="VideoDetectionServer",
    description="MCP Server to control and query the YOLO video detection system."
)

# --- MCP Tools ---

@mcp_server.tool()
def start_detection_system() -> str:
    """
    Starts the video detection and tracking system.
    Ensures the system is not already running before starting.
    """
    with start_stop_lock:
        system = get_detection_system()
        if system.is_running():
            logger.warning("start_detection_system: System is already running.")
            return "Detection system is already running."
        try:
            logger.info("MCP Tool: Attempting to start DetectionSystem...")
            # Run start in a separate thread to avoid blocking MCP?
            # For now, keep it simple, but be aware it might block.
            system.start()
            # Add a small delay and check if it actually started
            import time
            time.sleep(1) # Give threads time to spin up
            if system.is_running():
                logger.info("MCP Tool: DetectionSystem started successfully.")
                return "Detection system started successfully."
            else:
                logger.error("MCP Tool: DetectionSystem failed to start.")
                return "Error: Detection system failed to start."
        except Exception as e:
            logger.exception("MCP Tool: Exception during detection system start.")
            return f"Error starting detection system: {e}"

@mcp_server.tool()
def stop_detection_system() -> str:
    """
    Stops the video detection and tracking system.
    Ensures the system is running before attempting to stop.
    """
    with start_stop_lock:
        system = get_detection_system()
        if not system.is_running():
            logger.warning("stop_detection_system: System is not running.")
            return "Detection system is not running."
        try:
            logger.info("MCP Tool: Attempting to stop DetectionSystem...")
            system.stop()
             # Add a small delay and check if it actually stopped
            import time
            time.sleep(1) # Give threads time to shut down
            if not system.is_running():
                logger.info("MCP Tool: DetectionSystem stopped successfully.")
                return "Detection system stopped successfully."
            else:
                 logger.error("MCP Tool: DetectionSystem failed to stop cleanly.")
                 # Attempt to force stop? For now, just report.
                 return "Warning: Detection system may not have stopped cleanly."

        except Exception as e:
            logger.exception("MCP Tool: Exception during detection system stop.")
            return f"Error stopping detection system: {e}"

@mcp_server.tool()
def get_system_status() -> dict:
    """
    Checks if the detection system's worker threads are currently running.
    Returns:
        dict: A dictionary containing the running status, e.g., {"is_running": True/False}.
    """
    try:
        system = get_detection_system()
        is_running = system.is_running()
        logger.info(f"MCP Tool: get_system_status called. Running: {is_running}")
        return {"is_running": is_running}
    except Exception as e:
        # Handle case where system hasn't been initialized yet
        logger.warning(f"MCP Tool: Error checking system status (perhaps not initialized?): {e}")
        return {"is_running": False, "error": str(e)}


@mcp_server.tool()
def get_current_detections() -> dict:
    """
    Retrieves the latest detection results (bounding boxes, class IDs, confidence)
    and the dimensions of the frame they were detected in.
    Returns:
        dict: A dictionary containing 'detections', 'frame_width', and 'frame_height'.
              Returns empty/null values if the system is not running or no detections are available.
    """
    system = get_detection_system()
    if not system or not system.is_running():
        logger.warning("MCP Tool: get_current_detections called but system is not running.")
        return {"detections": [], "frame_width": None, "frame_height": None, "message": "System not running"}
    try:
        data = system.get_current_detections_data()
        logger.info(f"MCP Tool: get_current_detections retrieved {len(data.get('detections', []))} detections.")
        # Ensure data is serializable (should be, based on DetectionSystem code)
        return data
    except Exception as e:
        logger.exception("MCP Tool: Exception retrieving current detections.")
        return {"error": f"Error retrieving detections: {e}"}

@mcp_server.tool()
def get_tracked_objects() -> dict:
    """
    Retrieves information about currently tracked objects (ID, class, last seen position).
    Returns:
        dict: A dictionary where keys are track IDs and values are object details.
              Returns an empty dictionary if the system is not running or no objects are tracked.
    """
    system = get_detection_system()
    if not system or not system.is_running():
        logger.warning("MCP Tool: get_tracked_objects called but system is not running.")
        return {"tracked_objects": {}, "message": "System not running"}
    try:
        tracked_info = system.get_tracked_objects_info()
        logger.info(f"MCP Tool: get_tracked_objects retrieved info for {len(tracked_info)} objects.")
        return {"tracked_objects": tracked_info} # Wrap in a dict for clarity
    except Exception as e:
        logger.exception("MCP Tool: Exception retrieving tracked objects.")
        return {"error": f"Error retrieving tracked objects: {e}"}

@mcp_server.tool()
def get_object_track_history() -> dict:
    """
    Retrieves the recent position history for tracked objects.
    Returns:
        dict: A dictionary where keys are track IDs and values are lists of past center points.
              Returns an empty dictionary if the system is not running or no history is available.
    """
    system = get_detection_system()
    if not system or not system.is_running():
        logger.warning("MCP Tool: get_object_track_history called but system is not running.")
        return {"track_history": {}, "message": "System not running"}
    try:
        history = system.get_track_history()
        logger.info(f"MCP Tool: get_object_track_history retrieved history for {len(history)} objects.")
        return {"track_history": history} # Wrap in a dict for clarity
    except Exception as e:
        logger.exception("MCP Tool: Exception retrieving track history.")
        return {"error": f"Error retrieving track history: {e}"}


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting MCP Server for Video Detection System...")
    try:
        # Initialize the system when the server starts
        get_detection_system()
        logger.info("Running MCP server...")
        # Run the server using stdio communication as per the blog post example
        mcp_server.run()
    except RuntimeError as e:
         logger.critical(f"Failed to start MCP Server due to DetectionSystem initialization failure: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred while running the MCP server.")
    finally:
        # Ensure detection system is stopped when MCP server exits
        logger.info("MCP Server shutting down. Stopping DetectionSystem if running...")
        if detection_system_instance and detection_system_instance.is_running():
            with start_stop_lock: # Use lock to prevent race condition with stop tool
                 if detection_system_instance.is_running(): # Check again inside lock
                    detection_system_instance.stop()
                    logger.info("DetectionSystem stopped during MCP server shutdown.")
        logger.info("MCP Server finished.")

