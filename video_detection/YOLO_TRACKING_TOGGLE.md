# YOLO Object Tracking Toggle Feature

## Overview
Added independent controls for YOLO object tracking and tracking visualization rendering:
1. **YOLO Tracking**: Controls whether objects are tracked across frames with persistent IDs
2. **Track Lines Rendering**: Controls whether track path lines are displayed (track IDs in labels are always shown when tracking is enabled)

**Default State**: 
- **Tracking is ON** (enabled by default) - Objects get persistent track IDs
- **Track Lines Rendering is OFF** (disabled by default) - Track path lines are not shown, but track IDs remain in labels

**Important**: Track IDs are always included in labels when tracking is enabled, regardless of the rendering flag. This is intentional to allow other modules to read the IDs from the annotated frames. The `render_tracking` flag only controls the visual track path lines.

## Changes Made

### 1. DetectionSystem (`detection_system.py`)

#### New State Variables
- `tracking_enabled`: Boolean flag (default: `True`)  - Controls YOLO tracking
- `tracking_lock`: Threading lock for safe state access
- `render_tracking_enabled`: Boolean flag (default: `False`) - Controls visualization rendering
- `render_tracking_lock`: Threading lock for safe state access

#### New Methods for Tracking Control
- `enable_tracking()`: Enable YOLO object tracking
- `disable_tracking()`: Disable YOLO object tracking  
- `toggle_tracking()`: Toggle tracking on/off
- `is_tracking_enabled()`: Check if tracking is enabled

#### New Methods for Rendering Control
- `enable_render_tracking()`: Enable tracking visualization rendering
- `disable_render_tracking()`: Disable tracking visualization rendering
- `toggle_render_tracking()`: Toggle rendering on/off
- `is_render_tracking_enabled()`: Check if rendering is enabled

#### Updated Methods
- `__init__()`: Initialize tracking state to True, rendering state to False
- `start()`: Pass `tracking_enabled` parameter to ObjectDetector
- `process_single_image()`: Respect tracking state when processing single images
  - Uses `model.track()` when tracking is enabled
  - Uses `model.predict()` when tracking is disabled

### 2. ObjectDetector (`object_detector.py`)

#### New Parameters
- `tracking_enabled`: Added to constructor (default: `True`)

#### New State Variables
- `tracking_enabled`: Stores tracking state
- `tracking_lock`: Threading lock for safe state access

#### New Methods
- `set_tracking_enabled(enabled)`: Update tracking state dynamically

#### Updated Methods
- `process_frame()`: 
  - Uses `model.track()` when tracking is enabled
  - Uses `model.predict()` when tracking is disabled
  - Handles detections with or without track IDs
  - Uses temporary IDs (`temp_0`, `temp_1`, etc.) when tracking is off

### 3. AnnotationWorker (`annotation_worker.py`)

#### New Parameters
- `is_render_tracking_enabled_func`: Callback to check rendering state

#### Updated Methods
- `_annotate_loop()`:
  - Conditionally renders track path lines based on `render_tracking_enabled`
  - Always shows track IDs in labels when tracking is enabled (independent of render flag)
  - Always renders bounding boxes and masks (independent of render flag)
  
**Note**: Track IDs remain in labels regardless of the render_tracking flag to allow other modules to read them from the annotated frames.

### 4. Flask API (`app.py`)

#### New Endpoints for YOLO Tracking Control

##### Toggle Tracking
- **POST** `/api/yolo_tracking/toggle`
- Toggles YOLO tracking on/off
- Returns: `{"yolo_tracking_enabled": true/false}`

##### Enable Tracking
- **POST** `/api/yolo_tracking/enable`
- Explicitly enables YOLO tracking
- Returns: `{"yolo_tracking_enabled": true}`

##### Disable Tracking
- **POST** `/api/yolo_tracking/disable`
- Explicitly disables YOLO tracking
- Returns: `{"yolo_tracking_enabled": false}`

##### Get Tracking Status
- **GET** `/api/yolo_tracking/status`
- Returns current tracking state
- Returns: `{"yolo_tracking_enabled": true/false}`

#### New Endpoints for Tracking Rendering Control

##### Toggle Rendering
- **POST** `/api/render_tracking/toggle`
- Toggles track path line rendering on/off
- Returns: `{"render_tracking_enabled": true/false}`

##### Enable Rendering
- **POST** `/api/render_tracking/enable`
- Explicitly enables track path line rendering
- Returns: `{"render_tracking_enabled": true}`

##### Disable Rendering
- **POST** `/api/render_tracking/disable`
- Explicitly disables track path line rendering
- Returns: `{"render_tracking_enabled": false}`

##### Get Rendering Status
- **GET** `/api/render_tracking/status`
- Returns current rendering state
- Returns: `{"render_tracking_enabled": true/false}`

**Note**: These endpoints only control track path lines. Track IDs in labels are always shown when tracking is enabled.

## Usage Examples

### YOLO Tracking Control

#### Enable Tracking via API
```bash
curl -X POST http://localhost:5000/api/yolo_tracking/enable
```

#### Disable Tracking via API
```bash
curl -X POST http://localhost:5000/api/yolo_tracking/disable
```

#### Toggle Tracking via API
```bash
curl -X POST http://localhost:5000/api/yolo_tracking/toggle
```

#### Check Tracking Status
```bash
curl -X GET http://localhost:5000/api/yolo_tracking/status
```

### Tracking Rendering Control

#### Enable Rendering via API
```bash
curl -X POST http://localhost:5000/api/render_tracking/enable
```

#### Disable Rendering via API
```bash
curl -X POST http://localhost:5000/api/render_tracking/disable
```

#### Toggle Rendering via API
```bash
curl -X POST http://localhost:5000/api/render_tracking/toggle
```

#### Check Rendering Status
```bash
curl -X GET http://localhost:5000/api/render_tracking/status
```

**Note**: These control track path lines only. Track IDs in labels always appear when tracking is enabled.

## Behavior Differences

### Configuration Matrix

| Tracking | Render Lines | Track IDs in Labels | Track Path Lines | Result |
|----------|--------------|---------------------|------------------|--------|
| **ON** (default) | **OFF** (default) | ✅ Always shown | ❌ Hidden | Objects tracked, IDs visible, no path lines |
| ON | ON | ✅ Always shown | ✅ Visible | Full tracking with visual paths and IDs |
| OFF | OFF | ❌ Not available | ❌ Hidden | Simple detection, no tracking elements |
| OFF | ON | ❌ Not available | ❌ Hidden | Simple detection (render flag ignored) |

**Key Point**: When tracking is enabled, track IDs are ALWAYS included in labels (e.g., "person (ID: 5)") regardless of the `render_tracking_enabled` flag. This allows other modules to read track IDs from the annotated frames.

### When Tracking is ENABLED (tracking_enabled = True) - DEFAULT
- Uses `model.track()` with persistence
- Objects get consistent track IDs across frames
- Track history is maintained internally
- Better for following objects over time, analytics, counting
- Slightly more computational overhead

### When Tracking is DISABLED (tracking_enabled = False)
- Uses `model.predict()` 
- No track IDs assigned (uses temporary IDs)
- No track history maintained
- Fresh detection each frame
- Lower computational overhead
- Better for simple detection tasks where object identity doesn't matter

### When Rendering is ENABLED (render_tracking_enabled = True)
- Track path lines are drawn showing object movement paths
- Track IDs remain in labels (they're always shown when tracking is enabled)
- Requires tracking to be enabled to have meaningful effect
- Useful for debugging, monitoring, and visualizing object movements

### When Rendering is DISABLED (render_tracking_enabled = False) - DEFAULT
- No track path lines drawn
- Track IDs still shown in labels (e.g., "person (ID: 5)")
- Cleaner visual output without path clutter
- Tracking data still available via API and in labels
- Better for production where other modules need to read IDs from frames

## Notes

- **Default Configuration**: Tracking ON, Rendering OFF
  - Provides tracking data with IDs in labels for other modules
  - Keeps visual output clean (no path lines)
  - Best of both worlds for most use cases
- **Track IDs are always shown in labels when tracking is enabled** - This is intentional to allow other modules to read IDs from annotated frames
- The `render_tracking` flag only controls track path lines, not track IDs in labels
- The tracking state is preserved across video source changes
- The rendering state is independent of tracking state
- Thread-safe implementation using locks
- The existing `/api/toggle_tracking` endpoint controls bounding box drawing, not YOLO tracking
- When tracking is disabled, detections still work but without persistent object tracking
- Temporary IDs (format: `temp_0`, `temp_1`, etc.) are used when tracking is off
- Rendering has no effect when tracking is disabled (no track data to render)

## Performance Considerations

### Tracking Enabled + Rendering Disabled (DEFAULT)
- Moderate processing (tracking overhead)
- Track IDs shown in labels for module integration
- No track path lines (cleaner output)
- Track data available via API
- **Recommended for production with module integration**

### Tracking Enabled + Rendering Enabled
- Moderate processing + rendering overhead
- Visual feedback with path lines and IDs
- Track data available via API and in labels
- Good for debugging, monitoring, and demonstrations

### Tracking Disabled
- Fastest processing (no tracking overhead)
- Lower memory usage (no track history)
- Simpler detection pipeline
- Good for scenarios where object identity doesn't matter
- Rendering flag has no effect

## Use Cases

### Use Tracking ON + Rendering OFF (Default)
- **Production environments with module integration**
- Other modules need to read track IDs from frames
- API-based integrations
- Cleaner UI without path line clutter
- Data collection and analytics
- Object counting and monitoring

### Use Tracking ON + Rendering ON
- Development and debugging
- Visual monitoring dashboards with path visualization
- Demonstrations showing object movement
- Quality assurance testing
- Understanding object behavior patterns

### Use Tracking OFF
- Simple object detection
- Performance-critical applications
- Scenarios where object identity is not needed
- Quick detection without history
