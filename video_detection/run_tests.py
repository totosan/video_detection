#!/usr/bin/env python3
"""
Test runner for the track ID filtering functionality.
"""
import unittest
import sys
import os

# Add the video_detection directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import test modules
from test_track_filtering import TestTrackIdFiltering, TestFilteringLogic
from test_api_endpoints import TestTrackIdFilteringAPI, TestObjectFilteringAPI


def run_tests():
    """Run all tests for the track ID filtering functionality."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTrackIdFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestFilteringLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestTrackIdFilteringAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestObjectFilteringAPI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
