// Constants and Configuration
const API_ENDPOINTS = {
    TRACKED_OBJECTS: "/api/tracked_objects",
    BACKEND_ANNOTATION_STATUS: "/api/backend_annotation/status",
    BACKEND_ANNOTATION_TOGGLE: "/api/backend_annotation/toggle",
    OBJECT_FILTER_GET: "/api/get_object_filter",
    OBJECT_FILTER_SET: "/api/set_object_filter",
    CURRENT_DETECTIONS: "/api/current_detections",
    TRACKING_STATUS: "/api/tracking_status",
    TRACKING_TOGGLE: "/api/toggle_tracking",
    TRACK_ID_FILTER_GET: "/api/get_track_id_filter",
    TRACK_ID_FILTER_SET: "/api/set_track_id_filter",
};

const UPDATE_INTERVALS = {
    TRACKED_OBJECTS: 1000,
    FILTER_STATUS: 5000,
};

// Global state
let originalFrameWidth = null;
let originalFrameHeight = null;
let currentObjectFilter = [];

// DOM Elements (to be cached on DOMContentLoaded)
let toggleBtn, statusSpan, debugRenderingContainer, videoFeed, canvas, ctx,
    objectFilterInput, setObjectFilterBtn, currentFilterStatusLabel,
    toggleTrackingBtn, trackingStatusSpan, trackedObjectsList,
    cameraSelectList, setVideoSourceBtn, videoSourceStatus, rtspUrlInput; // Added rtspUrlInput

// Generic Utility Functions
function updateToggleButtoState(buttonElement, statusElement, isEnabled, enabledText, disabledText) {
    if (!buttonElement) return;
    if (isEnabled) {
        buttonElement.classList.add("button-active");
        buttonElement.textContent = enabledText;
    } else {
        buttonElement.classList.remove("button-active");
        buttonElement.textContent = disabledText;
    }
    if (statusElement && (statusElement.textContent.startsWith("Status: Toggling") || statusElement.textContent.startsWith("Status: Error"))) {
        statusElement.textContent = ""; // Clear temporary status
    }
}

// --- Debug Rendering Toggle ---
async function fetchDebugRenderingStatus() {
    if (!toggleBtn) return;
    try {
        const response = await fetch(API_ENDPOINTS.BACKEND_ANNOTATION_STATUS);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        updateToggleButtoState(toggleBtn, statusSpan, data.backend_annotation_enabled, "Hide Debug Rendering", "Show Debug Rendering");

        if (debugRenderingContainer) {
            if (data.backend_annotation_enabled) {
                debugRenderingContainer.classList.remove("hidden");
            } else {
                debugRenderingContainer.classList.add("hidden");
            }
        }
    } catch (error) {
        console.error("Error fetching debug rendering status:", error);
        if (statusSpan) statusSpan.textContent = "Status: Error";
        if (toggleBtn) toggleBtn.classList.remove("button-active");
    }
}

function setupDebugRenderingToggle() {
    if (!toggleBtn) return;
    toggleBtn.addEventListener("click", async () => {
        try {
            if (statusSpan) statusSpan.textContent = "Status: Toggling...";
            const response = await fetch(API_ENDPOINTS.BACKEND_ANNOTATION_TOGGLE, { method: "POST" });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            updateToggleButtoState(toggleBtn, statusSpan, data.backend_annotation_enabled, "Hide Debug Rendering", "Show Debug Rendering");
            if (debugRenderingContainer) {
                if (data.backend_annotation_enabled) {
                    debugRenderingContainer.classList.remove("hidden");
                } else {
                    debugRenderingContainer.classList.add("hidden");
                }
            }
        } catch (error) {
            console.error("Error toggling debug rendering:", error);
            if (statusSpan) statusSpan.textContent = "Status: Error";
            fetchDebugRenderingStatus(); // Re-fetch to ensure correct state
        }
    });
}

// --- Client-Side Detection Drawing ---
function resizeCanvas() {
    if (videoFeed && canvas) {
        canvas.width = videoFeed.clientWidth;
        canvas.height = videoFeed.clientHeight;
    }
}

async function fetchObjectFilterForInput() { // Primarily populates the input and global var
    try {
        const response = await fetch(API_ENDPOINTS.OBJECT_FILTER_GET);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        currentObjectFilter = data.object_filter || [];
        if (objectFilterInput) {
            objectFilterInput.value = currentObjectFilter.join(",");
        }
    } catch (error) {
        console.error("Error fetching object filter for input:", error);
    }
}

async function updateCurrentFilterStatusLabel() { // Primarily updates the status label, also refreshes global var
    try {
        const trackIdResponse = await fetch(API_ENDPOINTS.TRACK_ID_FILTER_GET);
        if (!trackIdResponse.ok) throw new Error(`HTTP error! status: ${trackIdResponse.status}`);
        const trackIdData = await trackIdResponse.json();

        const objectFilterResponse = await fetch(API_ENDPOINTS.OBJECT_FILTER_GET);
        if (!objectFilterResponse.ok) throw new Error(`HTTP error! status: ${objectFilterResponse.status}`);
        const objectFilterData = await objectFilterResponse.json();

        const trackIdFilter = trackIdData.track_id_filter;
        const objectFilter = objectFilterData.object_filter || [];

        const filterStatusLabel = document.getElementById("currentFilterStatusLabel");
        if (filterStatusLabel) {
            if (trackIdFilter !== null) {
                filterStatusLabel.textContent = `Track ID: ${trackIdFilter}`;
            } else if (objectFilter.length > 0) {
                filterStatusLabel.textContent = `Labels: ${objectFilter.join(", ")}`;
            } else {
                filterStatusLabel.textContent = "None (all objects shown)";
            }
        }
    } catch (error) {
        console.error("Error updating filter status label:", error);
        const filterStatusLabel = document.getElementById("currentFilterStatusLabel");
        if (filterStatusLabel) filterStatusLabel.textContent = "Error loading status";
    }
}

async function setObjectFilter() {
    if (!objectFilterInput) return;
    const filterValue = objectFilterInput.value.split(",").map(item => item.trim()).filter(item => item);
    try {
        const response = await fetch(API_ENDPOINTS.OBJECT_FILTER_SET, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ object_filter: filterValue })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        console.log("Object filter set to:", data.object_filter);
        currentObjectFilter = data.object_filter || []; // Update global filter
        updateCurrentFilterStatusLabel(); // Refresh the displayed active filter status
        displayFeedback("Object filter set successfully.");
    } catch (error) {
        console.error("Error setting object filter:", error);
        displayFeedback("Failed to set object filter.", true);
    }
}

async function displayFeedback(message, isError = false) {
    const feedbackElement = document.getElementById("feedbackMessage");
    if (feedbackElement) {
        feedbackElement.textContent = message;
        feedbackElement.style.color = isError ? "red" : "green";
        feedbackElement.style.display = "block";
        setTimeout(() => {
            feedbackElement.style.display = "none";
        }, 3000); // Hide after 3 seconds
    }
}

async function setTrackIdFilter() {
    const trackIdInput = document.getElementById("trackIdFilterInput");
    if (!trackIdInput) return;

    const trackIdValue = trackIdInput.value.trim();
    try {
        const response = await fetch(API_ENDPOINTS.TRACK_ID_FILTER_SET, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ track_id: trackIdValue ? parseInt(trackIdValue, 10) : null })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        console.log("Track ID filter set to:", data.track_id_filter);
        updateCurrentFilterStatusLabel();
        displayFeedback("Track ID filter set successfully.");
    } catch (error) {
        console.error("Error setting track ID filter:", error);
        displayFeedback("Failed to set Track ID filter.", true);
    }
}

async function clearFilters() {
    try {
        const trackIdResponse = await fetch(API_ENDPOINTS.TRACK_ID_FILTER_SET, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ track_id: null })
        });
        if (!trackIdResponse.ok) throw new Error(`HTTP error! status: ${trackIdResponse.status}`);

        const objectFilterResponse = await fetch(API_ENDPOINTS.OBJECT_FILTER_SET, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ object_filter: [] })
        });
        if (!objectFilterResponse.ok) throw new Error(`HTTP error! status: ${objectFilterResponse.status}`);

        console.log("All filters cleared.");
        updateCurrentFilterStatusLabel();
        displayFeedback("Filters cleared successfully.");
    } catch (error) {
        console.error("Error clearing filters:", error);
        displayFeedback("Failed to clear filters.", true);
    }
}

function setupObjectFilterControls() {
    if (setObjectFilterBtn) {
        setObjectFilterBtn.addEventListener("click", setObjectFilter);
    }
}

function setupTrackIdFilterControls() {
    const setTrackIdFilterBtn = document.getElementById("setTrackIdFilterBtn");
    const clearFilterBtn = document.getElementById("clearFilterBtn");

    if (setTrackIdFilterBtn) {
        setTrackIdFilterBtn.addEventListener("click", setTrackIdFilter);
    }

    if (clearFilterBtn) {
        clearFilterBtn.addEventListener("click", clearFilters);
    }
}

async function fetchAndDrawDetections() {
    if (!videoFeed || !canvas || !ctx) {
        requestAnimationFrame(fetchAndDrawDetections); // Keep trying if elements not ready
        return;
    }

    if (!videoFeed.complete || videoFeed.naturalWidth === 0) {
        requestAnimationFrame(fetchAndDrawDetections); // Wait for image to load
        return;
    }

    if (canvas.width !== videoFeed.clientWidth || canvas.height !== videoFeed.clientHeight) {
        resizeCanvas();
    }

    try {
        const response = await fetch(API_ENDPOINTS.CURRENT_DETECTIONS);
        if (!response.ok) {
            console.error("Failed to fetch detections:", response.status, response.statusText);
            try {
                const errorData = await response.json();
                console.error("Error data from API:", errorData);
            } catch (e) {
                // Ignore if error response is not JSON
            }
            requestAnimationFrame(fetchAndDrawDetections); // Try again on next frame
            return;
        }
        const data = await response.json();

        console.log("Received detections data:", JSON.stringify(data, null, 2)); // Log the full data structure

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Validate the structure of the received data
        if (!data || typeof data !== 'object') {
            console.error("API response data is missing or not an object.");
            requestAnimationFrame(fetchAndDrawDetections);
            return;
        }

        if (!Array.isArray(data.detections)) {
            console.error("API response 'data.detections' is missing or not an array.");
            requestAnimationFrame(fetchAndDrawDetections);
            return;
        }
        
        // data.frame_shape is expected to be [height, width] or null
        
        if (data.detections.length === 0) {
            // console.log("No detections in this frame."); // Optional: less verbose log
            requestAnimationFrame(fetchAndDrawDetections); // Continue loop even if no detections
            return;
        }

        // Update original frame dimensions if available and not yet set
        if ((!originalFrameWidth || !originalFrameHeight) && 
            data.frame_shape && 
            Array.isArray(data.frame_shape) && 
            data.frame_shape.length === 2) {
            
            originalFrameHeight = data.frame_shape[0]; // height
            originalFrameWidth = data.frame_shape[1];  // width

            if (originalFrameHeight <= 0 || originalFrameWidth <= 0) {
                console.warn(`Received frame_shape with zero or negative dimension: [${originalFrameHeight}, ${originalFrameWidth}]. Will use canvas size as fallback for scaling.`);
                originalFrameHeight = null; // Reset to allow fallback
                originalFrameWidth = null;  // Reset to allow fallback
            }
        }
        
        // Determine scaling factors. Fallback to canvas dimensions if original dimensions are unknown or invalid.
        const baseWidthForScale = (originalFrameWidth && originalFrameWidth > 0) ? originalFrameWidth : canvas.width;
        const baseHeightForScale = (originalFrameHeight && originalFrameHeight > 0) ? originalFrameHeight : canvas.height;

        // Prevent division by zero if base dimensions are still zero (e.g. canvas not rendered yet, or invalid originalFrame values)
        const scaleX = canvas.width / (baseWidthForScale || 1);
        const scaleY = canvas.height / (baseHeightForScale || 1);

        data.detections.forEach(det => {
            // Use the global currentObjectFilter
            if (currentObjectFilter.length > 0 && !currentObjectFilter.includes(det.label)) {
                return; // Skip if filter is active and label doesn\'t match
            }

            const [x1, y1, x2, y2] = det.box; // Still useful for label positioning
            const label = det.label || "unknown";
            const color = det.color ? `rgb(${det.color[0]}, ${det.color[1]}, ${det.color[2]})` : "red";
            const trackId = det.track_id || "unknown";

            const canvasX1 = x1 * scaleX;
            const canvasY1 = y1 * scaleY;
            // const canvasW = (x2 - x1) * scaleX; // Not directly used for masks, but good for context
            // const canvasH = (y2 - y1) * scaleY; // Not directly used for masks

            // Check for segmentation mask data
            // Assuming det.mask_points is an array of [x,y] normalized to 0-1 range
            if (det.mask_points && Array.isArray(det.mask_points) && det.mask_points.length > 0) {
                ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ', 0.5)'); // Semi-transparent fill
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;

                ctx.beginPath();
                // Scale normalized mask points by canvas dimensions directly
                ctx.moveTo(det.mask_points[0][0] * canvas.width, det.mask_points[0][1] * canvas.height);
                for (let i = 1; i < det.mask_points.length; i++) {
                    if (Array.isArray(det.mask_points[i]) && det.mask_points[i].length === 2) {
                        ctx.lineTo(det.mask_points[i][0] * canvas.width, det.mask_points[i][1] * canvas.height);
                    } else {
                        console.warn("Invalid point in mask_points array:", det.mask_points[i]);
                    }
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
            } else {
                // Fallback to drawing bounding box if no mask points or if mask_points is invalid
                // Ensure det.box is valid before trying to draw
                if (det.box && Array.isArray(det.box) && det.box.length === 4) {
                    const [x1, y1, x2, y2] = det.box;
                    const rectX = x1 * scaleX;
                    const rectY = y1 * scaleY;
                    const rectW = (x2 - x1) * scaleX;
                    const rectH = (y2 - y1) * scaleY;
                    
                    // Only draw if width and height are positive
                    if (rectW > 0 && rectH > 0) {
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 2;
                        ctx.strokeRect(rectX, rectY, rectW, rectH);
                    }
                } else {
                    console.warn("Fallback to bounding box: det.box is invalid", det.box);
                }
            }

            // Draw label (position based on the top-left of the bounding box)
            // Ensure det.box is valid for label positioning as well
            if (det.box && Array.isArray(det.box) && det.box.length === 4) {
                const [x1, y1, , ] = det.box; // Only need x1, y1 for label anchor
                const labelAnchorX = x1 * scaleX;
                const labelAnchorY = y1 * scaleY;

                ctx.fillStyle = color;
                const text = `${label} (ID: ${trackId})`;
                ctx.font = "12px Arial";
                const textMetrics = ctx.measureText(text);
                const textHeight = 12; // Approximate height for "12px Arial"
                
                // Ensure label is within canvas bounds
                let labelX = labelAnchorX;
                let labelY = labelAnchorY - textHeight - 4;
                if (labelY < 0) labelY = labelAnchorY + textHeight + 4; // If too high, draw below
                if (labelX + textMetrics.width + 4 > canvas.width) labelX = canvas.width - textMetrics.width - 4; // Adjust if too wide
                if (labelX < 0) labelX = 0;


                ctx.fillRect(labelX, labelY, textMetrics.width + 4, textHeight + 4);
                ctx.fillStyle = "white";
                ctx.fillText(text, labelX + 2, labelY + textHeight); // Adjusted y for fillText
            }
        });
    } catch (error) {
        console.error("Error fetching or drawing detections:", error);
    }
    requestAnimationFrame(fetchAndDrawDetections); // Continue the animation loop
}

// --- Camera Selection --- 
async function fetchAvailableCameras() {
    console.log("fetchAvailableCameras called");
    if (!cameraSelectList) {
        console.error("cameraSelectList element not found");
        return;
    }
    try {
        const response = await fetch("/api/cams");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const cameras = await response.json();
        console.log("Available cameras:", cameras);

        cameraSelectList.innerHTML = ''; // Clear existing options

        if (cameras.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "No cameras found";
            cameraSelectList.appendChild(option);
            console.log("No cameras found");
            return;
        }

        cameras.forEach(cam => {
            const option = document.createElement("option");
            // The value should be what the backend expects (index or RTSP URL)
            // Assuming 'index' for device cameras, and 'name' might be the RTSP URL or a descriptor
            // The backend's change_video_source will handle parsing this.
            option.value = cam.index; // Assuming index is the primary identifier for local cameras
            option.textContent = `${cam.name} (Index: ${cam.index}, ${cam.width}x${cam.height})`;
            cameraSelectList.appendChild(option);
        });
        console.log("Camera list populated successfully");
    } catch (error) {
        console.error("Error fetching available cameras:", error);
        if (cameraSelectList) {
            cameraSelectList.innerHTML = '<option value="">Error loading cameras</option>';
        }
        if (videoSourceStatus) videoSourceStatus.textContent = "Error loading cameras.";
    }
}

async function setSelectedVideoSource() {
    console.log("setSelectedVideoSource function called");
    if (!setVideoSourceBtn || !videoSourceStatus) {
        console.error("Required elements not found: setVideoSourceBtn or videoSourceStatus");
        return; // cameraSelectList and rtspUrlInput checked below
    }

    let selectedSourceIdentifier = "";
    const rtspValue = rtspUrlInput ? rtspUrlInput.value.trim() : "";
    console.log("RTSP input value:", rtspValue);

    if (rtspValue) {
        selectedSourceIdentifier = rtspValue;
        if (cameraSelectList) cameraSelectList.value = ""; // Clear dropdown selection
        console.log("Using RTSP URL from input:", selectedSourceIdentifier);
    } else if (cameraSelectList && cameraSelectList.value) {
        selectedSourceIdentifier = cameraSelectList.value;
        console.log("Using selected camera from dropdown:", selectedSourceIdentifier);
    } else {
        console.warn("No source selected");
        videoSourceStatus.textContent = "Please select a camera or enter an RTSP URL.";
        return;
    }

    videoSourceStatus.textContent = `Changing source to ${selectedSourceIdentifier}...`;
    setVideoSourceBtn.disabled = true;
    if (cameraSelectList) cameraSelectList.disabled = true;
    if (rtspUrlInput) rtspUrlInput.disabled = true;

    try {
        const response = await fetch("/api/selected_videosource", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source_identifier: selectedSourceIdentifier })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        // Modify success message based on RTSP URL content
        let successMessage = data.message || "Video source changed successfully!";
        if (rtspValue && rtspValue.includes("@")) { // Check if using RTSP input and it contains "@"
            successMessage = "Successfully changed video source to RTSP Source.";
        } else if (rtspValue) { // Using RTSP input but no "@"
             successMessage = `Successfully changed video source to ${selectedSourceIdentifier}`;
        } else { // Using dropdown
            successMessage = `Successfully changed video source to ${selectedSourceIdentifier}`;
        }
        videoSourceStatus.textContent = successMessage;
        console.log("Video source change successful:", data);

        // Clear the input that was used or the other one
        if (rtspValue) {
            if (rtspUrlInput) rtspUrlInput.value = ""; // Clear RTSP input if it was used
        } else {
            // If dropdown was used, no need to clear it here, 
            // but good to clear RTSP input if user typed something then selected dropdown
            if (rtspUrlInput) rtspUrlInput.value = ""; 
        }

        originalFrameWidth = null;
        originalFrameHeight = null;
        if (videoFeed) {
            // Force reload of the video feed image to reflect the new source
            // A common way is to append a meaningless query string that changes
            const currentSrc = videoFeed.src.split("?")[0];
            videoFeed.src = `${currentSrc}?t=${new Date().getTime()}`;
        }
        if (canvas && ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }


    } catch (error) {
        console.error("Error setting video source:", error);
        videoSourceStatus.textContent = `Error: ${error.message || "Failed to change source."}`;
    }
    setVideoSourceBtn.disabled = false;
    if (cameraSelectList) cameraSelectList.disabled = false;
    if (rtspUrlInput) rtspUrlInput.disabled = false;
}

function setupCameraControls() {
    console.log("setupCameraControls called");
    console.log("setVideoSourceBtn element:", setVideoSourceBtn);
    if (setVideoSourceBtn) {
        setVideoSourceBtn.addEventListener("click", setSelectedVideoSource);
        console.log("Event listener added to setVideoSourceBtn");
    } else {
        console.error("setVideoSourceBtn element not found");
    }
}

// --- Updated Tracked Objects List ---
function updateTrackedObjectsList() {
    if (!trackedObjectsList) return;

    // Fetch current detections instead of filtered tracked_objects to always display items
    fetch(API_ENDPOINTS.CURRENT_DETECTIONS)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            const fragment = document.createDocumentFragment();
            const detections = Array.isArray(data.detections) ? data.detections : [];
            if (detections.length === 0) {
                const item = document.createElement("li");
                item.textContent = "No objects detected.";
                fragment.appendChild(item);
            } else {
                detections.forEach(det => {
                    const listItem = document.createElement("li");
                    listItem.className = "list-group-item";
                    listItem.textContent = `ID: ${det.id}, Name: ${det.name}`;
                    listItem.style.cursor = "pointer";
                    listItem.addEventListener("click", () => {
                        if (det.name && objectFilterInput) {
                            objectFilterInput.value = det.name;
                            setObjectFilter();
                        }
                    });
                    fragment.appendChild(listItem);
                });
            }
            trackedObjectsList.innerHTML = "";
            trackedObjectsList.appendChild(fragment);
        })
        .catch(error => {
            console.error("Error fetching tracked objects:", error);
            if (trackedObjectsList) { // Check if list element exists
                 trackedObjectsList.innerHTML = "<li>Error loading tracked objects. Check console.</li>";
            }
        });
}

// --- Tracking Toggle ---
async function fetchTrackingStatus() {
    if (!toggleTrackingBtn) return;
    try {
        const response = await fetch(API_ENDPOINTS.TRACKING_STATUS);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        updateToggleButtoState(toggleTrackingBtn, trackingStatusSpan, data.tracking_enabled, "Disable Tracking", "Enable Tracking");
    } catch (error) {
        console.error("Error fetching tracking status:", error);
        if (trackingStatusSpan) trackingStatusSpan.textContent = "Status: Error";
        if (toggleTrackingBtn) toggleTrackingBtn.classList.remove("button-active");
    }
}

function setupTrackingToggle() {
    if (!toggleTrackingBtn) return;
    toggleTrackingBtn.addEventListener("click", async () => {
        try {
            if (trackingStatusSpan) trackingStatusSpan.textContent = "Status: Toggling...";
            const response = await fetch(API_ENDPOINTS.TRACKING_TOGGLE, { method: "POST" });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            updateToggleButtoState(toggleTrackingBtn, trackingStatusSpan, data.tracking_enabled, "Disable Tracking", "Enable Tracking");
        } catch (error) {
            console.error("Error toggling tracking:", error);
            if (trackingStatusSpan) trackingStatusSpan.textContent = "Status: Error";
            fetchTrackingStatus(); // Re-fetch to ensure correct state
        }
    });
}

// --- Initialization ---
function initializeApp() {
    // Cache DOM elements
    toggleBtn = document.getElementById("toggleDebugRenderingBtn");
    debugRenderingContainer = document.getElementById("debugRenderingContainer");
    videoFeed = document.getElementById("videoFeed");
    canvas = document.getElementById("detectionCanvas");
    if (canvas) { // Ensure canvas exists before getting context
        ctx = canvas.getContext("2d");
    }
    objectFilterInput = document.getElementById("objectFilterInput");
    setObjectFilterBtn = document.getElementById("setObjectFilterBtn");
    currentFilterStatusLabel = document.getElementById("currentFilterStatusLabel");
    toggleTrackingBtn = document.getElementById("toggleTrackingBtn");
    trackingStatusSpan = document.getElementById("trackingStatus");
    trackedObjectsList = document.getElementById("tracked-objects-list");

    // New camera control elements
    cameraSelectList = document.getElementById("cameraSelectList");
    rtspUrlInput = document.getElementById("rtspUrlInput"); // Cache RTSP input
    setVideoSourceBtn = document.getElementById("setVideoSourceBtn");
    videoSourceStatus = document.getElementById("videoSourceStatus");

    // Setup event listeners and initial state
    if (videoFeed) {
        videoFeed.onload = () => {
            // console.log("Video feed image loaded or reloaded.");
            resizeCanvas();
        };
        // Handle cases where the image might already be cached/loaded
        if (videoFeed.complete && videoFeed.naturalWidth !== 0) {
            //  console.log("Video feed image already complete.");
             resizeCanvas();
        }
    }
    window.addEventListener("resize", resizeCanvas);
    
    // Initial setup calls
    fetchDebugRenderingStatus();
    fetchObjectFilterForInput(); // Get initial filter state for the input box
    updateCurrentFilterStatusLabel(); // Get initial filter state for the status label
    fetchAvailableCameras();
    fetchTrackingStatus();

    // Setup event listeners
    setupDebugRenderingToggle();
    setupObjectFilterControls();
    setupTrackIdFilterControls();
    setupTrackingToggle();
    setupCameraControls();

    // Start the drawing loop
    requestAnimationFrame(fetchAndDrawDetections);
}

// Initialize app on DOMContentLoaded
document.addEventListener("DOMContentLoaded", initializeApp);

// Debug function to test camera source selection
function testCameraButton() {
    console.log("Testing camera button...");
    const btn = document.getElementById("setVideoSourceBtn");
    console.log("Button found:", btn);
    if (btn) {
        console.log("Button onclick:", btn.onclick);
        console.log("Button addEventListener count:", btn.getEventListeners ? btn.getEventListeners('click').length : 'getEventListeners not available');
    }
    
    const select = document.getElementById("cameraSelectList");
    console.log("Select element found:", select);
    if (select) {
        console.log("Select options:", select.options.length);
    }
}

// Call test function after a delay to ensure DOM is ready
setTimeout(testCameraButton, 2000);
