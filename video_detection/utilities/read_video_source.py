import cv2
import logging
import platform
import os
import subprocess # Added
import json       # Added

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_macos_camera_names():
    """
    Attempts to get camera names on macOS using system_profiler.
    Returns a list of camera names.
    """
    names = []
    try:
        cmd = ["system_profiler", "SPCameraDataType", "-json"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10) # Added timeout
        if process.returncode == 0:
            data = json.loads(stdout)
            if "SPCameraDataType" in data:
                for cam_data in data["SPCameraDataType"]:
                    # The name is usually under _name key
                    if "_name" in cam_data:
                        names.append(cam_data["_name"])
                    else: # Fallback if _name is not present but it's a camera entry
                        names.append("Unnamed macOS Camera") 
            if not names:
                logger.info("system_profiler found camera data but no specific names.")
        else:
            logger.warning(f"system_profiler command failed with error: {stderr.decode('utf-8', errors='ignore')}")
    except subprocess.TimeoutExpired:
        logger.warning("system_profiler command timed out.")
        if process:
            process.kill() # Ensure process is killed if it times out
            process.communicate() # Wait for process to terminate
    except FileNotFoundError:
        logger.warning("system_profiler command not found.")
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON output from system_profiler.")
    except Exception as e:
        logger.error(f"Error getting macOS camera names: {e}")
    return names

def find_available_cameras(max_cameras_to_check=10):
    """
    Checks for available cameras and logs their indices, names (OS-specific), and basic properties.
    Returns a list of dictionaries, each containing info about an available camera.
    """
    logger.info("Starting camera detection...")
    available_cameras_info = []
    
    macos_camera_names = []
    if platform.system() == "Darwin":
        logger.info("On macOS, attempting to get camera names via system_profiler.")
        macos_camera_names = get_macos_camera_names()
        if macos_camera_names:
            logger.info(f"system_profiler identified camera names: {macos_camera_names}")
        else:
            logger.info("system_profiler did not return any camera names or failed.")

    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        # if platform.system() == "Linux":
        #     cap = cv2.VideoCapture(i, cv2.CAP_V4L2) # Example for V4L2 on Linux
        # elif platform.system() == "Darwin":
        #     cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) # AVFoundation is default on macOS
        # else:
        #     cap = cv2.VideoCapture(i)


        if cap is None or not cap.isOpened():
            logger.debug(f"No camera found at index {i}.")
        else:
            device_name = f"Unknown Camera {i}" # Default name

            if platform.system() == "Linux":
                name_file_path = f"/sys/class/video4linux/video{i}/name"
                try:
                    if os.path.exists(name_file_path):
                        with open(name_file_path, 'r') as f:
                            device_name_from_file = f.read().strip()
                        if device_name_from_file:
                           device_name = device_name_from_file
                           logger.info(f"Camera found at index {i}: '{device_name}' (from {name_file_path})")
                        else:
                           logger.info(f"Camera found at index {i}. Name file '{name_file_path}' was empty. Using default name: '{device_name}'.")
                    else:
                        logger.info(f"Camera found at index {i}. Name file not found: '{name_file_path}'. Using default name: '{device_name}'.")
                except Exception as e:
                    logger.warning(f"Could not read name for camera {i} from '{name_file_path}': {e}. Using default name: '{device_name}'.")
            
            elif platform.system() == "Darwin":
                if i < len(macos_camera_names) and macos_camera_names[i]:
                    device_name = macos_camera_names[i]
                    logger.info(f"Camera found at index {i}. Matched with system_profiler name: '{device_name}'.")
                else:
                    logger.info(f"Camera found at index {i}. No corresponding name from system_profiler (or list exhausted). Using default name: '{device_name}'.")
            
            else: # Other OS
                logger.info(f"Camera found at index {i}. OS: {platform.system()}. Detailed name retrieval not implemented for this OS. Using default name: '{device_name}'.")

            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cam_info = {
                "index": i,
                "name": device_name,
                "width": int(width) if width else 0,
                "height": int(height) if height else 0,
                "fps": fps if fps else 0.0
            }
            logger.info(f"  Details: Index={cam_info['index']}, Name='{cam_info['name']}', Resolution={cam_info['width']}x{cam_info['height']}, FPS={cam_info['fps']:.2f}")
            available_cameras_info.append(cam_info)
            
            cap.release()
            logger.debug(f"Released camera at index {i}.")
    
    if not available_cameras_info:
        logger.info(f"No cameras found after checking up to index {max_cameras_to_check-1}.")
    else:
        logger.info(f"Found {len(available_cameras_info)} available camera(s):")
        for cam_info in available_cameras_info:
            logger.info(f"  - Index: {cam_info['index']}, Name: '{cam_info['name']}', Resolution: {cam_info['width']}x{cam_info['height']}, FPS: {cam_info['fps']:.2f}")
    return available_cameras_info

if __name__ == "__main__":
    find_available_cameras()
