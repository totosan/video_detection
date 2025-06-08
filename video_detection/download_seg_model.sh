#!/bin/bash
# Script to download YOLOv11n-seg model

MODEL="yolov11n-seg.pt"
URL="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov11n-seg.pt"

echo "Downloading $MODEL from Ultralytics assets..."
echo "This may take a while depending on your internet connection..."

# Try using curl if available
if command -v curl &> /dev/null; then
    curl -L -o "$MODEL" "$URL"
# Otherwise try wget
elif command -v wget &> /dev/null; then
    wget -O "$MODEL" "$URL"
else
    # If neither curl nor wget is available, try Python
    python3 -c "
import urllib.request
import sys
import os

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f'\rDownloading: {percent}% [{int(count * block_size / 1024 / 1024)} MB]')
    sys.stdout.flush()

try:
    print(f'Downloading {os.path.basename(\"$URL\")} from {\"$URL\"}...')
    urllib.request.urlretrieve(\"$URL\", \"$MODEL\", reporthook=download_progress)
    print(f'\nDownload complete! Model saved as {\"$MODEL\"}')
except Exception as e:
    print(f'\nError downloading model: {e}')
    sys.exit(1)
"
fi

if [ $? -eq 0 ]; then
    echo "Download complete. Model saved as $MODEL"
    exit 0
else
    echo "Download failed. Please check your internet connection or try downloading manually from: $URL"
    exit 1
fi
