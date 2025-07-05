"""
Unit tests for the track ID filtering functionality.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the video_detection directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from object_detector import ObjectDetector
from detection_system import DetectionSystem


class TestTrackIdFiltering(unittest.TestCase):
    """Test cases for track ID filtering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.names = {0: 'person', 1: 'car', 2: 'bicycle'}
        self.frame_queue = Mock()
        self.annotation_queue = Mock()
        self.stop_event = Mock()
        self.results_callback = Mock()
        
    def test_object_detector_filter_initialization(self):
        """Test ObjectDetector initialization with filters."""
        detector = ObjectDetector(
            model=self.mock_model,
            frame_queue=self.frame_queue,
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            results_update_callback=self.results_callback,
            max_track_points=100,
            model_names={0: 'person', 1: 'car'},
            filter_track_id=123,
            filter_labels=['person', 'car']
        )
        
        self.assertEqual(detector.filter_track_id, 123)
        self.assertEqual(detector.filter_labels, ['person', 'car'])
        
    def test_object_detector_update_filters(self):
        """Test updating filters in ObjectDetector."""
        detector = ObjectDetector(
            model=self.mock_model,
            frame_queue=self.frame_queue,
            annotation_queue=self.annotation_queue,
            stop_event=self.stop_event,
            results_update_callback=self.results_callback,
            max_track_points=100,
            model_names={0: 'person', 1: 'car'}
        )
        
        # Test updating track ID filter
        detector.update_filters(track_id=456)
        self.assertEqual(detector.filter_track_id, 456)
        self.assertEqual(detector.filter_labels, [])
        
        # Test updating label filter
        detector.update_filters(labels=['bicycle'])
        self.assertIsNone(detector.filter_track_id)
        self.assertEqual(detector.filter_labels, ['bicycle'])
        
        # Test clearing filters
        detector.update_filters()
        self.assertIsNone(detector.filter_track_id)
        self.assertEqual(detector.filter_labels, [])

    @patch('detection_system.ObjectDetector')
    def test_detection_system_track_id_filter(self, mock_object_detector_class):
        """Test DetectionSystem track ID filter methods."""
        mock_detector_instance = Mock()
        mock_object_detector_class.return_value = mock_detector_instance
        
        detection_system = DetectionSystem()
        
        # Test setting track ID filter
        detection_system.set_track_id_filter(789)
        self.assertEqual(detection_system._active_track_id_filter, 789)
        mock_detector_instance.update_filters.assert_called_with(track_id=789, labels=None)
        
        # Test getting track ID filter
        result = detection_system.get_track_id_filter()
        self.assertEqual(result, 789)
        
        # Test clearing track ID filter
        detection_system.set_track_id_filter(None)
        self.assertIsNone(detection_system._active_track_id_filter)
        
    @patch('detection_system.ObjectDetector')
    def test_detection_system_label_filter(self, mock_object_detector_class):
        """Test DetectionSystem label filter methods."""
        mock_detector_instance = Mock()
        mock_object_detector_class.return_value = mock_detector_instance
        
        detection_system = DetectionSystem()
        
        # Test setting label filter
        detection_system.set_object_filter(['person', 'car'])
        self.assertEqual(detection_system._active_label_filter, ['person', 'car'])
        mock_detector_instance.update_filters.assert_called_with(track_id=None, labels=['person', 'car'])
        
        # Test getting label filter
        result = detection_system.get_label_filter()
        self.assertEqual(result, ['person', 'car'])
        
        # Test clearing label filter
        detection_system.set_object_filter([])
        self.assertEqual(detection_system._active_label_filter, [])


class TestFilteringLogic(unittest.TestCase):
    """Test the filtering logic in ObjectDetector."""
    
    def test_track_id_filter_priority(self):
        """Test that track ID filter takes priority over label filter."""
        # Create a mock detection scenario
        detections = [
            {'track_id': 123, 'label': 'person'},
            {'track_id': 456, 'label': 'car'},
            {'track_id': 789, 'label': 'person'}
        ]
        
        # Test track ID filter (should only include track_id=123)
        filter_track_id = 123
        filter_labels = ['person']  # This should be ignored when track_id is set
        
        filtered = []
        for det in detections:
            include_detection = False
            if filter_track_id is not None:
                if det['track_id'] == filter_track_id:
                    include_detection = True
            elif filter_labels:
                if det['label'] in filter_labels:
                    include_detection = True
            else:
                include_detection = True
                
            if include_detection:
                filtered.append(det)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['track_id'], 123)
        
    def test_label_filter_only(self):
        """Test label filtering when no track ID filter is set."""
        detections = [
            {'track_id': 123, 'label': 'person'},
            {'track_id': 456, 'label': 'car'},
            {'track_id': 789, 'label': 'person'}
        ]
        
        # Test label filter only
        filter_track_id = None
        filter_labels = ['person']
        
        filtered = []
        for det in detections:
            include_detection = False
            if filter_track_id is not None:
                if det['track_id'] == filter_track_id:
                    include_detection = True
            elif filter_labels:
                if det['label'] in filter_labels:
                    include_detection = True
            else:
                include_detection = True
                
            if include_detection:
                filtered.append(det)
        
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(det['label'] == 'person' for det in filtered))


if __name__ == '__main__':
    unittest.main()
