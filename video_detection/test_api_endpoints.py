"""
Integration tests for the track ID filtering API endpoints.
"""
import unittest
from unittest.mock import Mock, patch
import json
import sys
import os

# Add the video_detection directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Mock the heavy dependencies before importing app
with patch('detection_system.DetectionSystem'), \
     patch('app.detection_system'):
    from app import app


class TestTrackIdFilteringAPI(unittest.TestCase):
    """Test cases for track ID filtering API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock the detection_system
        self.mock_detection_system = Mock()
        app.detection_system = self.mock_detection_system

    def test_set_track_id_filter_valid(self):
        """Test setting a valid track ID filter."""
        self.mock_detection_system.set_track_id_filter.return_value = None
        
        response = self.client.post('/api/set_track_id_filter',
                                  data=json.dumps({'track_id': 123}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['track_id_filter'], 123)
        self.mock_detection_system.set_track_id_filter.assert_called_once_with(123)

    def test_set_track_id_filter_null(self):
        """Test clearing the track ID filter."""
        self.mock_detection_system.set_track_id_filter.return_value = None
        
        response = self.client.post('/api/set_track_id_filter',
                                  data=json.dumps({'track_id': None}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsNone(data['track_id_filter'])
        self.mock_detection_system.set_track_id_filter.assert_called_once_with(None)

    def test_set_track_id_filter_missing_field(self):
        """Test setting track ID filter with missing field."""
        response = self.client.post('/api/set_track_id_filter',
                                  data=json.dumps({}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('track_id', data['error'])

    def test_get_track_id_filter(self):
        """Test getting the current track ID filter."""
        self.mock_detection_system.get_track_id_filter.return_value = 456
        
        response = self.client.get('/api/get_track_id_filter')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['track_id_filter'], 456)
        self.mock_detection_system.get_track_id_filter.assert_called_once()

    def test_get_track_id_filter_none(self):
        """Test getting track ID filter when none is set."""
        self.mock_detection_system.get_track_id_filter.return_value = None
        
        response = self.client.get('/api/get_track_id_filter')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsNone(data['track_id_filter'])

    def test_api_error_handling(self):
        """Test API error handling when detection system fails."""
        self.mock_detection_system.set_track_id_filter.side_effect = Exception("Test error")
        
        response = self.client.post('/api/set_track_id_filter',
                                  data=json.dumps({'track_id': 123}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('details', data)


class TestObjectFilteringAPI(unittest.TestCase):
    """Test cases for object filtering API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock the detection_system
        self.mock_detection_system = Mock()
        app.detection_system = self.mock_detection_system

    def test_set_object_filter_valid(self):
        """Test setting a valid object filter."""
        self.mock_detection_system.set_object_filter.return_value = None
        
        response = self.client.post('/api/set_object_filter',
                                  data=json.dumps({'object_filter': ['person', 'car']}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['object_filter'], ['person', 'car'])
        self.mock_detection_system.set_object_filter.assert_called_once_with(['person', 'car'])

    def test_get_object_filter(self):
        """Test getting the current object filter."""
        self.mock_detection_system.get_label_filter.return_value = ['bicycle']
        
        response = self.client.get('/api/get_object_filter')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['object_filter'], ['bicycle'])
        self.mock_detection_system.get_label_filter.assert_called_once()


if __name__ == '__main__':
    unittest.main()
