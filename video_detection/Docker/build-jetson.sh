#!/bin/bash

# Build script for NVIDIA Jetson Nano Docker image
# This script builds a Docker image optimized for Jetson Nano ARM64 architecture

set -e

# Configuration
IMAGE_NAME="video-detection"
TAG="jetson-nano"
DOCKERFILE_JETSON="Dockerfile.jetson"
DOCKERFILE_CI="Dockerfile.jetson-ci"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Docker image for NVIDIA Jetson Nano...${NC}"

# Ask user which Dockerfile to use
echo "Which Dockerfile would you like to use?"
echo "1) Dockerfile.jetson (For real Jetson hardware with L4T base)"
echo "2) Dockerfile.jetson-ci (For CI/emulation with Ubuntu base)"
read -p "Enter your choice (1-2): " choice

case $choice in
    1)
        DOCKERFILE=$DOCKERFILE_JETSON
        echo -e "${YELLOW}Using $DOCKERFILE for real Jetson hardware${NC}"
        ;;
    2)
        DOCKERFILE=$DOCKERFILE_CI
        TAG="jetson-nano-ci"
        echo -e "${YELLOW}Using $DOCKERFILE for CI/emulation${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Using default: $DOCKERFILE_JETSON${NC}"
        DOCKERFILE=$DOCKERFILE_JETSON
        ;;
esac

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: $DOCKERFILE not found!${NC}"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building image: ${IMAGE_NAME}:${TAG}${NC}"
docker build \
    -f "$DOCKERFILE" \
    -t "${IMAGE_NAME}:${TAG}" \
    --platform linux/arm64 \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    ../

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}Image: ${IMAGE_NAME}:${TAG}${NC}"
    
    # Display image info
    echo -e "${YELLOW}Image details:${NC}"
    docker images "${IMAGE_NAME}:${TAG}"
    
    echo -e "${YELLOW}To run the container on Jetson Nano:${NC}"
    echo "docker run --runtime nvidia --rm -p 8082:8082 -p 5555:5555 -p 5556:5556 ${IMAGE_NAME}:${TAG}"
    
    echo -e "${YELLOW}To run with GPU support and device access:${NC}"
    echo "docker run --runtime nvidia --rm -p 8082:8082 -p 5555:5555 -p 5556:5556 --device /dev/video0:/dev/video0 ${IMAGE_NAME}:${TAG}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
