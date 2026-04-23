# Jetson Nano Docker Deployment Guide

This guide explains how to build and run the video detection application on NVIDIA Jetson Nano devices.

## Prerequisites

### Jetson Nano Setup
1. **JetPack 4.6.1** (recommended) - includes:
   - Ubuntu 18.04 LTS
   - CUDA 10.2
   - cuDNN 8.2
   - TensorRT 8.0
   - OpenCV 4.1.1

2. **Docker with NVIDIA runtime**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   
   # Install NVIDIA Docker runtime
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Docker Compose** (optional):
   ```bash
   sudo pip3 install docker-compose
   ```

## Building the Image

### Option 1: Using the build script (Recommended)
```bash
cd Docker/
./build-jetson.sh
```

### Option 2: Manual build
```bash
docker build -f Dockerfile.jetson -t video-detection:jetson-nano --platform linux/arm64 ../
```

### Option 3: Using Docker Compose
```bash
docker-compose -f docker-compose.jetson.yml build
```

## Running the Container

### Basic Run
```bash
docker run --runtime nvidia --rm -p 8082:8082 -p 5555:5555 -p 5556:5556 video-detection:jetson-nano
```

### With Camera Access
```bash
docker run --runtime nvidia --rm -p 8082:8082 -p 5555:5555 -p 5556:5556 \
  --device /dev/video0:/dev/video0 \
  video-detection:jetson-nano
```

### With Persistent Storage
```bash
docker run --runtime nvidia --rm -p 8082:8082 -p 5555:5555 -p 5556:5556 \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/../Snapshots:/video_detection/Snapshots \
  video-detection:jetson-nano
```

### Using Docker Compose (Production)
```bash
docker-compose -f docker-compose.jetson.yml up -d
```

## Performance Optimization

### Memory Management
Jetson Nano has limited RAM (4GB). Consider these optimizations:

1. **Swap Configuration**:
   ```bash
   # Increase swap space
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Docker Memory Limits**:
   ```bash
   docker run --runtime nvidia --memory=3g --memory-swap=6g ...
   ```

### GPU Optimization
1. **Max Performance Mode**:
   ```bash
   sudo nvpmodel -m 0  # Max performance
   sudo jetson_clocks   # Max clocks
   ```

2. **Monitor GPU Usage**:
   ```bash
   sudo tegrastats
   ```

## Troubleshooting

### Common Issues

1. **CUDA Not Available**:
   - Ensure `--runtime nvidia` is used
   - Check: `nvidia-smi` works on host
   - Verify JetPack installation

2. **Memory Issues**:
   - Enable swap space
   - Reduce model complexity
   - Use smaller batch sizes

3. **Camera Access**:
   - Check camera device: `ls /dev/video*`
   - Ensure proper permissions: `sudo chmod 666 /dev/video0`
   - Test camera: `v4l2-ctl --list-devices`

4. **Build Failures**:
   - Ensure sufficient disk space (8GB+ free)
   - Check Docker daemon memory limits
   - Use `--no-cache` flag if needed

### Performance Monitoring
```bash
# System resources
htop

# GPU utilization
sudo tegrastats

# Docker container stats
docker stats video-detection-jetson
```

## Architecture-Specific Considerations

### ARM64 vs x86_64 Differences
- Uses `nvcr.io/nvidia/l4t-base:r32.7.1` instead of standard Python base
- Python 3.6 (JetPack limitation) vs Python 3.11
- ARM64-specific PyTorch wheels
- Different OpenCV compilation
- Longer build times due to ARM compilation

### Multi-Architecture Support
This setup allows you to maintain separate optimized builds:
- `Dockerfile.intel` - x86_64 systems
- `Dockerfile.jetson` - Jetson Nano (ARM64)
- `Dockerfile.intel-optimized` - Intel with optimizations

## API Endpoints

Once running, the application provides:
- **Web Interface**: http://jetson-nano-ip:8082
- **API Status**: http://jetson-nano-ip:8082/api/status
- **Detection API**: http://jetson-nano-ip:8082/api/detect
- **ZeroMQ Endpoints**: 5555 (images), 5556 (results)

## Next Steps

1. **Production Deployment**: Use docker-compose with proper networking
2. **Load Balancing**: Multiple Jetson devices with reverse proxy
3. **Monitoring**: Add Prometheus/Grafana for metrics
4. **Auto-scaling**: Based on GPU utilization
