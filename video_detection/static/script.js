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
    toggleTrackingBtn, trackingStatusSpan, trackedObjectsList;

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
        const response = await fetch(API_ENDPOINTS.OBJECT_FILTER_GET); // Fetch to get the latest
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        currentObjectFilter = data.object_filter || []; // Update global state

        if (currentFilterStatusLabel) {
            if (currentObjectFilter.length > 0) {
                currentFilterStatusLabel.textContent = currentObjectFilter.join(", ");
            } else {
                currentFilterStatusLabel.textContent = "None (all objects shown)";
            }
        }
    } catch (error) {
        console.error("Error fetching current filter status for label:", error);
        if (currentFilterStatusLabel) currentFilterStatusLabel.textContent = "Error loading status";
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
    } catch (error) {
        console.error("Error setting object filter:", error);
    }
}

function setupObjectFilterControls() {
    if (setObjectFilterBtn) {
        setObjectFilterBtn.addEventListener("click", setObjectFilter);
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
            console.error("Failed to fetch detections:", response.statusText);
            requestAnimationFrame(fetchAndDrawDetections); // Try again on next frame
            return;
        }
        const data = await response.json();

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!data.detections || data.detections.length === 0) {
            requestAnimationFrame(fetchAndDrawDetections); // Continue loop even if no detections
            return;
        }

        // Ensure original frame dimensions are set (once)
        if (!originalFrameWidth || !originalFrameHeight) {
            originalFrameWidth = data.frame_width;
            originalFrameHeight = data.frame_height;
        }
        
        // Prevent division by zero if frame dimensions are not yet available or are zero
        const scaleX = canvas.width / (originalFrameWidth || 1);
        const scaleY = canvas.height / (originalFrameHeight || 1);

        data.detections.forEach(det => {
            // Use the global currentObjectFilter
            if (currentObjectFilter.length > 0 && !currentObjectFilter.includes(det.label)) {
                return; // Skip if filter is active and label doesn"t match
            }

            const [x1, y1, x2, y2] = det.box;
            const label = det.label || "unknown";
            const color = det.color ? `rgb(${det.color[0]}, ${det.color[1]}, ${det.color[2]})` : "red";

            const canvasX1 = x1 * scaleX;
            const canvasY1 = y1 * scaleY;
            const canvasW = (x2 - x1) * scaleX;
            const canvasH = (y2 - y1) * scaleY;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(canvasX1, canvasY1, canvasW, canvasH);

            ctx.fillStyle = color;
            const text = `${label} (ID: ${det.track_id || "unknown"})`;
            ctx.font = "12px Arial";
            const textMetrics = ctx.measureText(text);
            const textHeight = 12; // Approximate height for "12px Arial"
            ctx.fillRect(canvasX1, canvasY1 - textHeight - 4, textMetrics.width + 4, textHeight + 4);

            ctx.fillStyle = "white";
            ctx.fillText(text, canvasX1 + 2, canvasY1 - 4);
        });
    } catch (error) {
        console.error("Error fetching or drawing detections:", error);
    }
    requestAnimationFrame(fetchAndDrawDetections); // Continue the animation loop
}

// --- Updated Tracked Objects List ---
function updateTrackedObjectsList() {
    if (!trackedObjectsList) return;

    fetch(API_ENDPOINTS.TRACKED_OBJECTS)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            const fragment = document.createDocumentFragment(); // Use DocumentFragment

            if (!Array.isArray(data)) {
                console.error("Tracked objects data is not an array:", data);
                const errorItem = document.createElement("li");
                errorItem.textContent = "Error: Invalid data format from server.";
                fragment.appendChild(errorItem);
            } else if (data.length === 0) {
                const noObjectsItem = document.createElement("li");
                noObjectsItem.textContent = "No objects tracked recently.";
                fragment.appendChild(noObjectsItem);
            } else {
                try {
                    data.sort((a, b) => { // Sort by time_since_seen (most recent first)
                        const timeA = parseFloat(a.time_since_seen);
                        const timeB = parseFloat(b.time_since_seen);
                        if (isNaN(timeA) && isNaN(timeB)) return 0;
                        if (isNaN(timeA)) return 1; // Push NaNs to the end
                        if (isNaN(timeB)) return -1;
                        return timeA - timeB; // Ascending sort
                    });
                } catch (e) {
                    console.error("Error sorting tracked objects:", e, data);
                    const errorItem = document.createElement("li");
                    errorItem.textContent = "Error: Could not sort object data.";
                    fragment.appendChild(errorItem);
                    // Clear list and append only the error
                    trackedObjectsList.innerHTML = "";
                    trackedObjectsList.appendChild(fragment);
                    return; 
                }
                
                const limitedData = data.slice(0, 5); // Limit to 5 most recent

                limitedData.forEach(obj => {
                    const listItem = document.createElement("li");
                    listItem.className = "list-group-item"; // Bootstrap class, ensure it"s defined in your CSS if used
                    listItem.style.cursor = "pointer";

                    if (obj.time_since_seen < 3.0) {
                        listItem.classList.add("recent");
                    } else {
                        listItem.classList.add("stale");
                    }

                    if (obj.detection_image) {
                        const img = document.createElement("img");
                        img.src = "data:image/jpeg;base64," + obj.detection_image;
                        img.alt = `Object ${obj.id}`;
                        img.style.width = "60px";
                        img.style.height = "60px";
                        img.style.marginRight = "10px";
                        listItem.appendChild(img);
                    }

                    // Ensure time_since_seen is a number before calling toFixed
                    const timeSinceSeenText = typeof obj.time_since_seen === "number" ? obj.time_since_seen.toFixed(1) : obj.time_since_seen;
                    const textNode = document.createTextNode(
                        `ID: ${obj.id}, Name: ${obj.name}, Seen: ${timeSinceSeenText}s ago`
                    );
                    listItem.appendChild(textNode);

                    listItem.addEventListener("click", () => {
                        if (obj.name && objectFilterInput) {
                            objectFilterInput.value = obj.name;
                            setObjectFilter(); // Apply the filter
                            console.log(`Filter set to "${obj.name}" by clicking tracked object.`);
                        }
                    });
                    fragment.appendChild(listItem);
                });
            }
            trackedObjectsList.innerHTML = ""; // Clear current list once
            trackedObjectsList.appendChild(fragment); // Append all new items
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
    
    setupDebugRenderingToggle();
    setupObjectFilterControls();
    setupTrackingToggle();

    // Fetch initial states
    fetchDebugRenderingStatus();
    fetchObjectFilterForInput();       // Populates input and currentObjectFilter
    updateCurrentFilterStatusLabel();  // Updates the label based on (potentially just fetched) currentObjectFilter
    fetchTrackingStatus();
    updateTrackedObjectsList();        // Initial call for tracked objects

    // Start loops
    requestAnimationFrame(fetchAndDrawDetections); // Start drawing loop

    setInterval(updateTrackedObjectsList, UPDATE_INTERVALS.TRACKED_OBJECTS);
    setInterval(updateCurrentFilterStatusLabel, UPDATE_INTERVALS.FILTER_STATUS); // Periodically update filter status label
}

// Wait for the DOM to be fully loaded before initializing
document.addEventListener("DOMContentLoaded", initializeApp);

// Remove obsolete function (if it was in the original HTML script block)
// function updateTrackedObjects() { /* ... */ } // This is now removed.
