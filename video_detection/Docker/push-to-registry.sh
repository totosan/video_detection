#!/bin/bash

# Manual push script for GitHub Container Registry
set -e

# Configuration
REGISTRY="ghcr.io"
OWNER="totosan"
IMAGE_NAME="video-detection"
VERSION=$(date +%Y%m%d-%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}GitHub Container Registry Manual Push Script${NC}"
echo "Registry: ${REGISTRY}"
echo "Owner: ${OWNER}"
echo "Image: ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo

# Check if logged in to registry
echo -e "${YELLOW}Checking registry authentication...${NC}"

# Try a simple registry check - if this fails, the push will fail anyway
echo "Testing registry access..."
if docker info >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker is running and ready${NC}"
else
    echo -e "${RED}✗ Docker is not running or accessible${NC}"
    exit 1
fi

echo "Note: If you haven't authenticated yet, use:"
echo "  gh auth token | docker login ghcr.io -u ${OWNER} --password-stdin"
echo

# Detect script location and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"
echo

# Function to build and push an image
build_and_push() {
    local variant="$1"
    local dockerfile="$2"
    
    echo -e "${YELLOW}Building and pushing ${variant} variant...${NC}"
    
    local full_image_name="${REGISTRY}/${OWNER}/${IMAGE_NAME}"
    local latest_tag="${full_image_name}:${variant}-latest"
    local version_tag="${full_image_name}:${variant}-${VERSION}"
    
    echo "Building ${dockerfile}..."
    docker build \
        --platform linux/amd64 \
        -f "Docker/${dockerfile}" \
        -t "${latest_tag}" \
        -t "${version_tag}" \
        --progress plain \
        .
    
    echo "Pushing ${latest_tag}..."
    docker push "${latest_tag}"
    
    echo "Pushing ${version_tag}..."
    docker push "${version_tag}"
    
    echo -e "${GREEN}✓ Successfully pushed ${variant} variant${NC}"
    echo "  - ${latest_tag}"
    echo "  - ${version_tag}"
    echo
}

# Build and push variants
echo -e "${YELLOW}Starting build and push process...${NC}"
echo

# Ask user which variants to build
echo "Which variants would you like to build and push?"
echo "1) Intel standard only"
echo "2) Intel optimized only"
echo "3) Both variants"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        build_and_push "intel" "Dockerfile.intel"
        ;;
    2)
        build_and_push "intel-optimized" "Dockerfile.intel-optimized"
        ;;
    3)
        build_and_push "intel" "Dockerfile.intel"
        build_and_push "intel-optimized" "Dockerfile.intel-optimized"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 All images pushed successfully to GitHub Container Registry!${NC}"
echo
echo "You can now pull your images with:"
echo "  docker pull ${REGISTRY}/${OWNER}/${IMAGE_NAME}:intel-latest"
echo "  docker pull ${REGISTRY}/${OWNER}/${IMAGE_NAME}:intel-optimized-latest"
echo
echo "Or view them at: https://github.com/${OWNER}?tab=packages&repo_name=${IMAGE_NAME}"
