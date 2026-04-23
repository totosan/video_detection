# Track ID Filtering Implementation

This document describes the track ID filtering functionality implemented in the YOLO-based video detection system.

## Overview

The system now supports filtering detected objects by both:
1. **Track ID**: Display only objects with a specific track ID
2. **Object Labels**: Display only objects with specific labels (existing functionality)

Track ID filtering takes priority over label filtering when both are set.

## Components Modified

### Backend Changes

#### 1. ObjectDetector (`object_detector.py`)
- Added `filter_track_id` and `filter_labels` parameters to constructor
- Implemented filtering logic in `_detect_objects()` method
- Added `update_filters()` method for runtime filter updates
- Filter priority: Track ID > Labels > No filter (show all)

#### 2. DetectionSystem (`detection_system.py`)
- Added `_active_track_id_filter` and `_active_label_filter` state variables
- Implemented filter management methods:
  - `set_track_id_filter(track_id)`
  - `get_track_id_filter()`
  - `set_object_filter(labels)` (updated)
  - `get_label_filter()` (updated)
- Filter state is reset when system stops/resets

#### 3. Flask API (`app.py`)
- Added new endpoints:
  - `POST /api/set_track_id_filter` - Set track ID filter
  - `GET /api/get_track_id_filter` - Get current track ID filter
- Updated existing endpoints to use new detection system methods
- Added proper error handling and validation

### Frontend Changes

#### 4. HTML Template (`templates/index.html`)
- Added Track ID filter input field and button
- Added "Select Closest Object" button for LLM workflows
- Added feedback message display area
- Added "Clear All Filters" button
- Added current filter status display

#### 5. JavaScript (`static/script.js`)
- Added track ID filter management functions
- Implemented "select closest object" logic (finds largest bounding box)
- Added user feedback system for success/error messages
- Updated filter status display to show active filter type
- Added proper error handling for API calls

## API Endpoints

### Track ID Filtering
```
POST /api/set_track_id_filter
Content-Type: application/json
{
  "track_id": 123  // or null to clear
}

GET /api/get_track_id_filter
Returns: {"track_id_filter": 123}  // or null
```

### Object Label Filtering (Updated)
```
POST /api/set_object_filter
Content-Type: application/json
{
  "object_filter": ["person", "car"]  // or [] to clear
}

GET /api/get_object_filter
Returns: {"object_filter": ["person", "car"]}
```

## User Interface

### Filter Controls
1. **Object Label Filter**: Comma-separated list of labels
2. **Track ID Filter**: Single numeric track ID
3. **Select Closest Object**: Automatically selects the object with the largest bounding box
4. **Clear All Filters**: Removes both track ID and label filters
5. **Current Filter Status**: Shows which filter is currently active

### Filter Priority
- Track ID filter takes precedence over label filter
- When track ID is set, label filter is ignored
- Status display shows the active filter type

## LLM Integration Workflow

The "Select Closest Object" feature enables LLM workflows:

1. LLM analyzes current detection data via `/api/current_detections`
2. LLM identifies the object with the largest bounding box (closest to camera)
3. LLM calls `selectClosestObject()` function or sets track ID filter directly
4. System displays only the selected object

## Testing

Tests are provided for:
- ObjectDetector filtering logic
- DetectionSystem filter management
- Flask API endpoints
- Filter priority and behavior

Run tests with:
```bash
cd /Users/toto/Projects/JetsonNano/ai-video-solution/video_detection
python run_tests.py
```

## Backward Compatibility

All changes are backward compatible:
- Existing label-based filtering continues to work
- Default behavior (no filters) shows all objects
- API responses maintain existing structure

## Error Handling

- Frontend displays success/error messages for filter operations
- API endpoints return appropriate HTTP status codes
- Invalid filter values are handled gracefully
- System continues to function if filtering fails

## Files Modified

1. `object_detector.py` - Core filtering logic
2. `detection_system.py` - Filter state management
3. `app.py` - API endpoints
4. `templates/index.html` - UI controls
5. `static/script.js` - Frontend logic

## Files Added

1. `test_track_filtering.py` - Unit tests for filtering logic
2. `test_api_endpoints.py` - Integration tests for API
3. `run_tests.py` - Test runner script
4. This documentation file

## Usage Examples

### Set Track ID Filter
```javascript
// Frontend JavaScript
fetch('/api/set_track_id_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({track_id: 123})
});
```

### Select Closest Object
```javascript
// Frontend JavaScript - automatically finds and filters closest object
selectClosestObject();
```

### Clear All Filters
```javascript
// Frontend JavaScript
clearFilters();
```
