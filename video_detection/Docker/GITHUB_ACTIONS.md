# GitHub Actions Multi-Architecture Container Build

This document explains the GitHub Actions workflow for building and publishing multi-architecture Docker images for the video detection system.

## 🔄 Workflow Overview

The GitHub Actions workflow (`.github/workflows/build-containers.yml`) now supports **three build targets**:

### 1. **Intel Standard** (`linux/amd64`)
- **Dockerfile**: `Dockerfile.intel`
- **Image Tag**: `ghcr.io/totosan/video-detection:intel-latest`
- **Platform**: `linux/amd64`

### 2. **Intel Optimized** (`linux/amd64`)
- **Dockerfile**: `Dockerfile.intel-optimized`
- **Image Tag**: `ghcr.io/totosan/video-detection:intel-optimized-latest`
- **Platform**: `linux/amd64`

### 3. **Jetson Nano** (`linux/arm64`) ⭐ **NEW**
- **Dockerfile**: `Dockerfile.jetson`
- **Image Tag**: `ghcr.io/totosan/video-detection:jetson-nano-latest`
- **Platform**: `linux/arm64`

## 🚀 Workflow Triggers

The workflow automatically runs on:

```yaml
on:
  push:
    branches:
      - main
      - segmentation_containerimages
    tags:
      - 'v*'
  pull_request:
    branches:
      - main
```

### 📦 Generated Images

#### For Branch Pushes:
- `ghcr.io/totosan/video-detection:main-intel`
- `ghcr.io/totosan/video-detection:main-intel-optimized`
- `ghcr.io/totosan/video-detection:main-jetson-nano`
- `ghcr.io/totosan/video-detection:intel-latest` (main branch only)
- `ghcr.io/totosan/video-detection:intel-optimized-latest` (main branch only)
- `ghcr.io/totosan/video-detection:jetson-nano-latest` (main branch only)
- `ghcr.io/totosan/video-detection:latest` (main branch, multi-arch manifest)

#### For Version Tags (e.g., `v1.2.3`):
- `ghcr.io/totosan/video-detection:v1.2.3-intel`
- `ghcr.io/totosan/video-detection:v1.2.3-intel-optimized`
- `ghcr.io/totosan/video-detection:v1.2.3-jetson-nano`
- `ghcr.io/totosan/video-detection:v1.2.3` (multi-arch manifest)

## 🏗️ Multi-Architecture Manifest

The workflow creates **multi-architecture manifests** that automatically serve the correct image based on the target platform:

```bash
# This automatically pulls the right architecture:
docker pull ghcr.io/totosan/video-detection:latest

# On x86_64: Gets intel-latest
# On ARM64: Gets jetson-nano-latest
```

## 📋 Workflow Jobs

### Job 1: `build-intel-standard`
```yaml
platforms: linux/amd64
file: ./video_detection/Docker/Dockerfile.intel
```

### Job 2: `build-intel-optimized`
```yaml
platforms: linux/amd64
file: ./video_detection/Docker/Dockerfile.intel-optimized
```

### Job 3: `build-jetson-nano` ⭐ **NEW**
```yaml
platforms: linux/arm64
file: ./video_detection/Docker/Dockerfile.jetson
```

### Job 4: `create-multi-arch-manifest`
```yaml
needs: [build-intel-standard, build-intel-optimized, build-jetson-nano]
# Creates unified manifests for latest and version tags
```

## 🔧 Local Development & Testing

### Manual Registry Push
Use the updated push script for local builds:

```bash
cd Docker/
./push-to-registry.sh

# Options:
# 1) Intel standard only
# 2) Intel optimized only
# 3) Jetson Nano only          ← NEW
# 4) Intel variants (both)
# 5) All variants (Intel + Jetson) ← NEW
```

### Authentication
First-time setup for GitHub Container Registry:

```bash
# Login with GitHub token
gh auth token | docker login ghcr.io -u totosan --password-stdin

# Or use personal access token
echo $GITHUB_TOKEN | docker login ghcr.io -u totosan --password-stdin
```

## 🎯 Usage Examples

### Pull Images by Architecture
```bash
# Intel x86_64 standard
docker pull ghcr.io/totosan/video-detection:intel-latest

# Intel x86_64 optimized
docker pull ghcr.io/totosan/video-detection:intel-optimized-latest

# Jetson Nano ARM64
docker pull ghcr.io/totosan/video-detection:jetson-nano-latest

# Multi-arch (auto-detects platform)
docker pull ghcr.io/totosan/video-detection:latest
```

### Run on Different Platforms
```bash
# On Intel/AMD x86_64 systems
docker run --rm -p 8082:8082 ghcr.io/totosan/video-detection:intel-latest

# On Jetson Nano
docker run --runtime nvidia --rm -p 8082:8082 \
  --device /dev/video0:/dev/video0 \
  ghcr.io/totosan/video-detection:jetson-nano-latest
```

## 🔍 Registry Access

All images are available at:
**https://github.com/totosan/video-detection/pkgs/container/video-detection**

### Permissions
- **Public**: Anyone can pull images
- **Private**: Requires GitHub authentication for pushes
- **Packages**: Managed through GitHub repository settings

## 📊 Build Status

You can monitor build status at:
**https://github.com/totosan/video-detection/actions**

### Build Artifacts
- **Build logs**: Available in Actions tab
- **Cache optimization**: GitHub Actions cache for faster rebuilds
- **Multi-platform support**: ARM64 and AMD64 builds run in parallel

## 🔄 Continuous Integration

The workflow ensures:
- ✅ **Consistent builds** across all platforms
- ✅ **Automated testing** through Docker health checks
- ✅ **Version management** via Git tags
- ✅ **Platform-specific optimizations**
- ✅ **Multi-architecture manifest creation**

## 🎉 Benefits

1. **Automatic Builds**: Push to main → automatic container builds
2. **Multi-Platform**: Single workflow supports x86_64 and ARM64
3. **Version Management**: Git tags create versioned releases
4. **Registry Integration**: Direct GitHub Container Registry publishing
5. **Platform Detection**: Multi-arch manifests serve correct architecture
6. **Optimized Caching**: Fast rebuilds using GitHub Actions cache
