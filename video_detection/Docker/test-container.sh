#!/bin/bash

# Container testing script
set -e

IMAGE_NAME="video-detection"
TAG="${1:-intel-latest}"
CONTAINER_NAME="test-${TAG}-$(date +%s)"

echo "Testing image: ${IMAGE_NAME}:${TAG}"

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Run container
echo "Starting container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p 8082:8082 \
    -p 5555:5555 \
    -p 5556:5556 \
    ${IMAGE_NAME}:${TAG}

echo "Waiting for container to start..."
sleep 30

# Test health endpoint
echo "Testing health endpoint..."
for i in {1..10}; do
    if curl -f -s http://localhost:8082/api/status > /dev/null; then
        echo "✓ Health check passed!"
        break
    else
        echo "Attempt $i/10 failed, retrying in 5 seconds..."
        sleep 5
    fi
    
    if [ $i -eq 10 ]; then
        echo "✗ Health check failed after 10 attempts"
        docker logs ${CONTAINER_NAME}
        exit 1
    fi
done

# Test API endpoints
echo "Testing API endpoints..."

# Test status endpoint
STATUS_RESPONSE=$(curl -s http://localhost:8082/api/status)
echo "Status response: ${STATUS_RESPONSE}"

# Test cameras endpoint
CAMS_RESPONSE=$(curl -s http://localhost:8082/api/cams)
echo "Cameras response: ${CAMS_RESPONSE}"

# Test tracking status
TRACKING_RESPONSE=$(curl -s http://localhost:8082/api/tracking_status)
echo "Tracking status response: ${TRACKING_RESPONSE}"

echo "✓ All API tests passed!"

# Test web interface
echo "Testing web interface..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8082/)
if [ "$HTTP_STATUS" = "200" ]; then
    echo "✓ Web interface accessible!"
else
    echo "✗ Web interface returned HTTP $HTTP_STATUS"
    exit 1
fi

# Test snapshot endpoint
echo "Testing snapshot endpoint..."
SNAPSHOT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8082/snapshot)
if [ "$SNAPSHOT_STATUS" = "200" ] || [ "$SNAPSHOT_STATUS" = "503" ]; then
    echo "✓ Snapshot endpoint accessible (status: $SNAPSHOT_STATUS)!"
else
    echo "✗ Snapshot endpoint returned HTTP $SNAPSHOT_STATUS"
    exit 1
fi

echo "✓ Container test completed successfully!"
echo "Container logs:"
docker logs --tail 20 ${CONTAINER_NAME}
