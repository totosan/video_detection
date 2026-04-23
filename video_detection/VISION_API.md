# Vision API Documentation

## Object Detection API

# Vision API Documentation

## Object Detection API

### Endpoint
```
POST /v1/vision/detect/object
```

### Description
Detects objects in uploaded images and returns their labels, confidence scores, and bounding boxes with comprehensive metadata.

### Request Format
- **Method**: POST
- **Content-Type**: multipart/form-data

### Input Parameters

#### Required Parameters
- **image** - The image file to analyze (supports common formats like JPG, PNG)

#### Optional Parameters
- **min_confidence** - Minimum confidence threshold for detections (float, default: 0.4)
  - Range: 0.0 to 1.0
  - Only objects with confidence above this threshold will be returned

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Other formats supported by OpenCV

### Response Format
```json
{
    "success": true,
    "message": "Found person, car, dog...",
    "count": 3,
    "predictions": [
        {
            "label": "person",
            "confidence": 0.85,
            "x_min": 100,
            "y_min": 150,
            "x_max": 300,
            "y_max": 450
        },
        {
            "label": "car",
            "confidence": 0.92,
            "x_min": 400,
            "y_min": 200,
            "x_max": 650,
            "y_max": 400
        }
    ],
    "processMs": 234,
    "inferenceMs": 156,
    "moduleId": "ObjectDetectionYOLOv11",
    "analysisRoundTripMs": 245,
    "timestampUTC": "Mon, 04 Aug 2025 12:34:56 GMT"
}
```

### Response Fields

#### Core Response
- **success** - Boolean indicating if the operation was successful
- **message** - Human-readable summary of detected objects
- **count** - Number of objects detected
- **predictions** - Array of detected objects

#### Each Prediction Object Contains
- **label** - Object class name (e.g., "person", "car", "dog", "bicycle")
- **confidence** - Detection confidence score (0.0 to 1.0)
- **x_min** - Left edge of bounding box (pixels)
- **y_min** - Top edge of bounding box (pixels)
- **x_max** - Right edge of bounding box (pixels)
- **y_max** - Bottom edge of bounding box (pixels)

#### Metadata (added by server)
- **processMs** - Total processing time in milliseconds
- **inferenceMs** - AI inference time in milliseconds
- **moduleId** - ID of the module that processed the request
- **analysisRoundTripMs** - Round-trip time for the analysis
- **timestampUTC** - UTC timestamp of the response

### JavaScript Example
```javascript
// Using with a file input element
const fileInput = document.getElementById('fileChooser');
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('min_confidence', '0.5');

fetch('http://localhost:32168/v1/vision/detection', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log(`Found ${data.count} objects: ${data.message}`);
        console.log(`Processing took ${data.processMs}ms`);
        
        data.predictions.forEach((prediction, index) => {
            console.log(`Object ${index + 1}: ${prediction.label} (${prediction.confidence.toFixed(3)})`);
            console.log(`  Location: (${prediction.x_min}, ${prediction.y_min}) to (${prediction.x_max}, ${prediction.y_max})`);
        });
    } else {
        console.error('Detection failed:', data.message);
    }
})
.catch(error => {
    console.error('Error:', error);
});
```

### Python Example
```python
import requests

# Upload an image file with custom confidence threshold
with open('path/to/image.jpg', 'rb') as image_file:
    files = {'image': image_file}
    data = {'min_confidence': '0.6'};
    
    response = requests.post('http://localhost:32168/v1/vision/detection', 
                           files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print(f"Found {result['count']} objects: {result['message']}")
            print(f"Processing took {result['processMs']}ms")
            
            for prediction in result['predictions']:
                print(f"Found {prediction['label']} with confidence {prediction['confidence']:.3f}")
                print(f"  Location: ({prediction['x_min']}, {prediction['y_min']}) to ({prediction['x_max']}, {prediction['y_max']})")
        else:
            print(f"Detection failed: {result['message']}")
    else:
        print(f"HTTP Error: {response.status_code}")
```

### cURL Example
```bash
# Basic request
curl -X POST 
  -F "image=@/path/to/image.jpg" 
  http://localhost:32168/v1/vision/detection

# With custom confidence threshold
curl -X POST 
  -F "image=@/path/to/image.jpg" 
  -F "min_confidence=0.6" 
  http://localhost:32168/v1/vision/detection
```

### Error Response Format
When `success` is `false`, the response will include error information:
```json
{
    "success": false,
    "message": "Error description",
    "count": 0,
    "predictions": [],
    "processMs": 123,
    "inferenceMs": 0,
    "moduleId": "ObjectDetectionYOLOv11",
    "analysisRoundTripMs": 123,
    "timestampUTC": "Mon, 04 Aug 2025 12:34:56 GMT"
}
```

### HTTP Status Codes
- **200 OK**: Request processed successfully (check `success` field for detection result)
- **400 Bad Request**: Missing image file or invalid parameters
- **500 Internal Server Error**: Processing error

### Supported Object Classes
The endpoint can detect common COCO dataset objects including:

- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, bird, horse, cow, sheep
- **Household items**: chair, couch, tv, laptop, mouse, keyboard
- **And many more** (80+ classes total from COCO dataset)

### Performance Notes
- Detection confidence threshold affects both accuracy and performance
- Lower thresholds (e.g., 0.3) return more objects but may include false positives
- Higher thresholds (e.g., 0.7) return fewer, more confident detections
- Processing time varies based on image size and complexity
- The API uses YOLOv11 model for object detection
- Supports both regular detection and segmentation models

### Testing
Use the provided `test_vision_api.py` script to test the API:

```bash
cd /path/to/video_detection
python test_vision_api.py
```

### Request Format
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**: Image file in the 'image' field

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Other formats supported by OpenCV

### Response Format
```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.98,
      "boundingBox": {
        "x": 123,
        "y": 45,
        "width": 56,
        "height": 78
      }
    },
    {
      "label": "dog",
      "confidence": 0.85,
      "boundingBox": {
        "x": 200,
        "y": 100,
        "width": 50,
        "height": 60
      }
    }
  ]
}
```

### Response Fields
- **objects**: Array of detected objects
  - **label**: String - The detected object class (e.g., "person", "car", "dog")
  - **confidence**: Float - Detection confidence score (0.0 to 1.0)
  - **boundingBox**: Object - Bounding box coordinates
    - **x**: Integer - Left coordinate (pixels)
    - **y**: Integer - Top coordinate (pixels)
    - **width**: Integer - Width in pixels
    - **height**: Integer - Height in pixels

### JavaScript Example
```javascript
// Using with a file input element
const fileInput = document.getElementById('fileChooser');
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:3000/v1/vision/detect/object', {
    method: "POST",
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Detected objects:', data.objects);
    data.objects.forEach((obj, index) => {
        console.log(`Object ${index + 1}:`, obj.label, `(${obj.confidence.toFixed(2)})`);
    });
})
.catch(error => {
    console.error('Error:', error);
});
```

### Python Example
```python
import requests

# Upload an image file
with open('path/to/image.jpg', 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post('http://localhost:3000/v1/vision/detect/object', files=files)
    
    if response.status_code == 200:
        result = response.json()
        for obj in result['objects']:
            print(f"Found {obj['label']} with confidence {obj['confidence']:.2f}")
            bbox = obj['boundingBox']
            print(f"  Location: ({bbox['x']}, {bbox['y']}) size: {bbox['width']}x{bbox['height']}")
    else:
        print(f"Error: {response.status_code}")
```

### cURL Example
```bash
curl -X POST \
  -F "image=@/path/to/image.jpg" \
  http://localhost:3000/v1/vision/detect/object
```

### Error Responses
- **400 Bad Request**: Missing image file or invalid image format
- **500 Internal Server Error**: Processing error

### Error Response Format
```json
{
  "error": "Error description"
}
```

### Notes
- The API uses YOLOv11 model for object detection
- Detection confidence threshold is set to 0.25
- The API can handle segmentation models if available (mask data included when present)
- Maximum image size depends on available memory and processing power
- Processing time varies based on image size and complexity

### Testing
Use the provided `test_vision_api.py` script to test the API:

```bash
cd /path/to/video_detection
python test_vision_api.py
```
