#!/bin/bash

# Test script for Jetson Nano Docker container
# This script tests the container functionality on Jetson Nano

set -e

# Configuration
IMAGE_NAME="video-detection:jetson-nano"
CONTAINER_NAME="test-video-detection-jetson"
TEST_PORT=8082

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing Jetson Nano Docker container...${NC}"

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo -e "${RED}Error: Image $IMAGE_NAME not found!${NC}"
    echo -e "${YELLOW}Please build the image first: ./build-jetson.sh${NC}"
    exit 1
fi

# Cleanup any existing test container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo -e "${YELLOW}Starting container...${NC}"

# Start container
docker run -d --runtime nvidia \
    --name $CONTAINER_NAME \
    -p $TEST_PORT:8082 \
    -p 5555:5555 \
    -p 5556:5556 \
    $IMAGE_NAME

# Wait for container to start
echo -e "${YELLOW}Waiting for container to start...${NC}"
sleep 30

# Test if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}✅ Container is running${NC}"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test API endpoint
echo -e "${YELLOW}Testing API endpoint...${NC}"
if curl -f -s http://localhost:$TEST_PORT/api/status > /dev/null; then
    echo -e "${GREEN}✅ API is responding${NC}"
else
    echo -e "${RED}❌ API is not responding${NC}"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
    exit 1
fi

# Test GPU access (if available)
echo -e "${YELLOW}Testing GPU access...${NC}"
GPU_TEST=$(docker exec $CONTAINER_NAME python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "GPU test failed")
echo -e "${GREEN}GPU Test Result: $GPU_TEST${NC}"

# Show container stats
echo -e "${YELLOW}Container statistics:${NC}"
docker stats $CONTAINER_NAME --no-stream

# Show logs
echo -e "${YELLOW}Recent container logs:${NC}"
docker logs --tail 20 $CONTAINER_NAME

# Cleanup
echo -e "${YELLOW}Cleaning up test container...${NC}"
docker rm -f $CONTAINER_NAME

echo -e "${GREEN}✅ Test completed successfully!${NC}"
echo -e "${GREEN}The Jetson Nano container is working properly.${NC}"
