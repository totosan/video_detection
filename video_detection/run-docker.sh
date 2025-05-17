#! /bin/bash

# Usage: ./run-docker.sh <video_source>
# Example: ./run-docker.sh /dev/video0
#          ./run-docker.sh "rtsp://user:pass@host:port/path"

if [ -z "$1" ]; then
  echo "Usage: $0 <video_source>"
  echo "Example: $0 /dev/video0"
  echo "         $0 \"rtsp://user:pass@host:port/path\""
  exit 1
fi

VIDEO_URL=$1

docker run --rm --runtime nvidia \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env VIDEO_URL="$VIDEO_URL" \
  --network host \
  --shm-size=8g \
  --volume /argus_socket:/tmp/argus_socket \
  --volume /enctune.conf:/etc/enctune.conf \
  --volume /nv_tegra_release:/etc/nv_tegra_release \
  --volume /nv_jetson_model:/tmp/nv_jetson_model \
  --volume /run/dbus:/var/run/dbus \
  --volume /run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
  --volume /run/docker.sock:/var/run/docker.sock \
  -v /localtime:/etc/localtime:ro \
  -v /timezone:/etc/timezone:ro \
  --device-cgroup-rule='c 189:* rmw' \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /jtop.sock:/run/jtop.sock \
  -v "$(pwd)":/ros_ws \
  -w /ros_ws \
  --name ros_detection \
  yolo_detection
