# Docker Container Images for Video Detection

This directory contains Docker configurations for building and running the video detection system on various platforms.

## Available Images

### 1. Intel x86_64 (Standard)
- **Dockerfile**: `Dockerfile.intel`
- **Build Script**: `build-intel.sh`
- **Target**: General Intel/AMD x86_64 processors
- **Features**: Standard Python environment with OpenCV and YOLO support

### 2. Intel x86_64 (Optimized)
- **Dockerfile**: `Dockerfile.intel-optimized`
- **Build Script**: `build-intel-optimized.sh`
- **Target**: Intel processors with CPU optimizations
- **Features**: 
  - OpenBLAS for optimized linear algebra
  - PyTorch CPU version for better performance
  - Optimized threading settings
  - CPU-specific optimizations

## Quick Start

### Using Build Scripts

1. **Build Intel Standard Image**:
   ```bash
   cd Docker
   chmod +x build-intel.sh
   ./build-intel.sh
   ```

2. **Build Intel Optimized Image**:
   ```bash
   cd Docker
   chmod +x build-intel-optimized.sh
   ./build-intel-optimized.sh
   ```

### Using Docker Compose

1. **Standard Intel Image**:
   ```bash
   docker-compose -f Docker/docker-compose.intel.yml up -d video-detection-intel
   ```

2. **Optimized Intel Image**:
   ```bash
   docker-compose -f Docker/docker-compose.intel.yml --profile optimized up -d video-detection-intel-optimized
   ```

### Manual Docker Build

```bash
# Standard Intel
docker build -f Docker/Dockerfile.intel -t video-detection:intel-latest .

# Optimized Intel
docker build -f Docker/Dockerfile.intel-optimized -t video-detection:intel-optimized-latest .
```

## Running Containers

### Basic Run Command
```bash
docker run -d \
  -p 8082:8082 \
  -p 5555:5555 \
  -p 5556:5556 \
  --name video-detection-intel \
  video-detection:intel-latest
```

### With Volume Mounts
```bash
docker run -d \
  -p 8082:8082 \
  -p 5555:5555 \
  -p 5556:5556 \
  -v $(pwd)/Snapshots:/video_detection/Snapshots \
  -v $(pwd)/static:/video_detection/static \
  --name video-detection-intel \
  video-detection:intel-latest
```

### With Custom Configuration
```bash
docker run -d \
  -p 8082:8082 \
  -p 5555:5555 \
  -p 5556:5556 \
  -e FLASK_RUN_HOST=0.0.0.0 \
  -e FLASK_RUN_PORT=8082 \
  -e OMP_NUM_THREADS=4 \
  --name video-detection-intel \
  video-detection:intel-optimized-latest
```

## Ports

- **8082**: Flask web interface
- **5555**: ZeroMQ image input socket
- **5556**: ZeroMQ results output socket

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_RUN_HOST` | `0.0.0.0` | Flask server host |
| `FLASK_RUN_PORT` | `8082` | Flask server port |
| `DETECTION_IMAGE_ENDPOINT` | `tcp://*:5555` | ZeroMQ image input endpoint |
| `DETECTION_RESULTS_ENDPOINT` | `tcp://*:5556` | ZeroMQ results endpoint |
| `OMP_NUM_THREADS` | `4` | OpenMP thread count |
| `MKL_NUM_THREADS` | `4` | Intel MKL thread count |
| `OPENBLAS_NUM_THREADS` | `4` | OpenBLAS thread count |

## Health Checks

All containers include health checks that monitor the `/api/status` endpoint:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start Period**: 60 seconds

## Performance Tuning

### For Intel Optimized Images
The optimized Intel image includes several performance enhancements:

1. **CPU Optimizations**: Uses OpenBLAS and optimized NumPy builds
2. **Threading**: Configured for optimal CPU utilization
3. **PyTorch CPU**: Uses CPU-optimized PyTorch for better inference performance
4. **Memory Management**: Optimized memory allocation patterns

### Recommended Settings
- **CPU Cores**: Set `OMP_NUM_THREADS` to match available CPU cores
- **Memory**: Allocate at least 1GB RAM, recommended 2GB for better performance
- **Storage**: SSD recommended for model loading and snapshot storage

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If ports 8082, 5555, or 5556 are in use, modify the port mappings
2. **Memory Issues**: Increase Docker memory limit if containers are killed
3. **Permission Issues**: Ensure Docker has proper permissions for volume mounts

### Debugging

1. **View Container Logs**:
   ```bash
   docker logs video-detection-intel
   ```

2. **Enter Container Shell**:
   ```bash
   docker exec -it video-detection-intel /bin/bash
   ```

3. **Health Check Status**:
   ```bash
   docker inspect video-detection-intel | grep -A 10 Health
   ```

## Security Considerations

- Containers run as non-root user (`appuser`) for security
- Only necessary ports are exposed
- No sensitive data should be stored in containers
- Use secrets management for production deployments

## Building for Production

For production use, consider:

1. **Multi-stage builds** to reduce image size
2. **Security scanning** of built images
3. **Resource limits** appropriate for your infrastructure
4. **Persistent storage** for snapshots and logs
5. **Load balancing** for multiple instances
6. **Monitoring and logging** integration

## Next Steps

This is the first step in creating container images for multiple platforms. Future additions will include:
- ARM64 support for Apple Silicon and ARM servers
- NVIDIA GPU support for accelerated inference
- Edge device optimizations (Raspberry Pi, Jetson, etc.)
- Kubernetes deployment manifests
