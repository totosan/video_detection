#!/bin/bash

# Jetson Nano startup script to handle PyTorch library loading issues

# Set up environment variables to fix PyTorch TLS issues
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD"

# Alternative library paths to try
GOMP_PATHS=(
    "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
    "/usr/lib/aarch64-linux-gnu/libgomp.so"
    "/usr/local/lib/libgomp.so.1"
    "/usr/local/lib/libgomp.so"
)

# Find the correct libgomp path
for path in "${GOMP_PATHS[@]}"; do
    if [ -f "$path" ]; then
        export LD_PRELOAD="$path:$LD_PRELOAD"
        echo "Using libgomp at: $path"
        break
    fi
done

# Print environment info for debugging
echo "======== Jetson Startup Debug Info ========"
echo "Architecture: $(uname -m)"
echo "LD_PRELOAD: $LD_PRELOAD"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Python version: $(python3 --version)"
echo "============================================"

# Try to import torch to verify it works
echo "Testing PyTorch import..."
python3 -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print('PyTorch import successful!')
except Exception as e:
    print(f'PyTorch import failed: {e}')
    import sys
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "PyTorch test passed, starting application..."
    # Start the main application
    exec python3 app.py "$@"
else
    echo "PyTorch test failed, cannot start application"
    exit 1
fi
