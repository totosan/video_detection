#!/bin/bash

# Build script for Intel x86_64 optimized Docker image
set -e

# Configuration
IMAGE_NAME="video-detection"
TAG_PREFIX="intel-optimized"
VERSION=$(date +%Y%m%d-%H%M%S)
DOCKERFILE="Dockerfile.intel-optimized"

# Detect the correct paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building Intel x86_64 optimized Docker image..."
echo "Image: ${IMAGE_NAME}:${TAG_PREFIX}-${VERSION}"
echo "Dockerfile: ${DOCKERFILE}"
echo "Script directory: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"

# Build the image from the project root
cd "$PROJECT_ROOT"
docker build \
    --no-cache \
    --platform linux/amd64 \
    -f ./Docker/${DOCKERFILE} \
    -t ${IMAGE_NAME}:${TAG_PREFIX}-${VERSION} \
    -t ${IMAGE_NAME}:${TAG_PREFIX}-latest \
    --progress plain \
    .

echo "Build completed successfully!"
echo "Image tags:"
echo "  - ${IMAGE_NAME}:${TAG_PREFIX}-${VERSION}"
echo "  - ${IMAGE_NAME}:${TAG_PREFIX}-latest"

# Optional: Test the image
read -p "Do you want to test the image? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing image..."
    docker run --rm -p 8082:8082 ${IMAGE_NAME}:${TAG_PREFIX}-latest &
    CONTAINER_PID=$!
    
    # Wait a few seconds for the container to start
    sleep 10
    
    # Test the health endpoint
    if curl -f http://localhost:8082/api/status; then
        echo "✓ Image test passed!"
    else
        echo "✗ Image test failed!"
    fi
    
    # Stop the test container
    kill $CONTAINER_PID 2>/dev/null || true
fi

echo "To run the container:"
echo "docker run -d -p 8082:8082 -p 5555:5555 -p 5556:5556 --name video-detection-intel-opt ${IMAGE_NAME}:${TAG_PREFIX}-latest"
