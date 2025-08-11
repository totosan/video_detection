#!/usr/bin/env python3
"""
Test script for the /v1/vision/detection API endpoint.
This script demonstrates how to call the API endpoint with an image file.
"""

import requests
import json
import os
from pathlib import Path

def test_vision_api(image_path, api_url="http://192.168.0.39:32168/v1/vision/detection", min_confidence=0.4):
    """
    Test the vision API with an image file.
    
    Args:
        image_path: Path to the image file
        api_url: URL of the API endpoint
        min_confidence: Minimum confidence threshold for detections
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        # Prepare the multipart form data
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            data = {'min_confidence': str(min_confidence)}
            
            print(f"Sending request to: {api_url}")
            print(f"Image file: {image_path}")
            print(f"Min confidence: {min_confidence}")
            
            # Make the POST request
            response = requests.post(api_url, files=files, data=data, timeout=30)
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    print(f"✅ Success! {result.get('message', '')}")
                    print(f"Found {result.get('count', 0)} objects:")
                    print(f"Process time: {result.get('processMs', 0)}ms")
                    print(f"Inference time: {result.get('inferenceMs', 0)}ms")
                    print(f"Module: {result.get('moduleId', 'Unknown')}")
                    print(f"Timestamp: {result.get('timestampUTC', 'Unknown')}")
                    
                    for i, prediction in enumerate(result.get('predictions', []), 1):
                        print(f"  Object {i}:")
                        print(f"    Label: {prediction.get('label')}")
                        print(f"    Confidence: {prediction.get('confidence'):.3f}")
                        print(f"    Bounding Box: x_min={prediction.get('x_min')}, y_min={prediction.get('y_min')}, x_max={prediction.get('x_max')}, y_max={prediction.get('y_max')}")
                else:
                    print(f"❌ Request failed: {result.get('message', 'Unknown error')}")
                
                return result
            else:
                print(f"Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data}")
                except:
                    print(f"Error response text: {response.text}")
                return None
                
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {api_url}")
        print("Make sure the video detection server is running on the expected port.")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_with_javascript_example():
    """
    Demonstrate the equivalent JavaScript code for calling the API.
    """
    js_code = '''
// JavaScript example (updated API):
const formData = new FormData();
formData.append('image', imageFile);
formData.append('min_confidence', '0.5');

fetch('http://localhost:32168/v1/vision/detection', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(data);
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

// Expected response format:
// {
//     "success": true,
//     "message": "Found person, car, dog...",
//     "count": 3,
//     "predictions": [
//         {
//             "label": "person",
//             "confidence": 0.85,
//             "x_min": 100,
//             "y_min": 150,
//             "x_max": 300,
//             "y_max": 450
//         }
//     ],
//     "processMs": 234,
//     "inferenceMs": 156,
//     "moduleId": "ObjectDetectionYOLOv11",
//     "analysisRoundTripMs": 245,
//     "timestampUTC": "Mon, 04 Aug 2025 12:34:56 GMT"
// }
'''
    print("JavaScript equivalent code:")
    print(js_code)

if __name__ == "__main__":
    # Look for test images in the Snapshots directory
    snapshots_dir = Path("Snapshots")
    if snapshots_dir.exists():
        # Find the first image file
        for image_file in snapshots_dir.glob("*.jpg"):
            print(f"Testing with image: {image_file}")
            
            # Test with different confidence levels
            print("\n=== Testing with min_confidence=0.3 ===")
            result1 = test_vision_api(str(image_file), min_confidence=0.3)
            
            print("\n=== Testing with min_confidence=0.6 ===")
            result2 = test_vision_api(str(image_file), min_confidence=0.6)
            
            break
        else:
            print("No .jpg files found in Snapshots directory")
    else:
        print("Snapshots directory not found")
        print("You can test with any image file by calling:")
        print("  python test_vision_api.py")
        print("Or test programmatically:")
        print("  result = test_vision_api('/path/to/image.jpg', min_confidence=0.5)")
    
    print("\n" + "="*60)
    test_with_javascript_example()
