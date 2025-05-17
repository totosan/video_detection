# run docker command
docker run --runtime nvidia --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -it --network host --shm-size=8g --volume /tmp/argus_socket:/tmp/argus_socket --volume /etc/enctune.conf:/etc/enctune.conf --volume /etc/nv_tegra_release:/etc/nv_tegra_release --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model --volume /var/run/dbus:/var/run/dbus --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket --volume /var/run/docker.sock:/var/run/docker.sock  -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --device /dev/snd --device /dev/bus/usb --device /dev/i2c-0 --device /dev/i2c-1 --device /dev/i2c-2 --device /dev/i2c-3 --device /dev/i2c-4 --device /dev/i2c-5 --device /dev/i2c-6 --device /dev/i2c-7 --device /dev/i2c-8 -v /run/jtop.sock:/run/jtop.sock -v $(pwd):/ros_ws -w /ros_ws --name ultralytics_yolo yolo_detection

# install CUDA 11 for Ubuntu 18.04
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_arm64.deb
apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
apt-get update
apt-get -y install cuda

## install the dependencies (if not already onboard)
sudo apt-get install python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
pip3 install future
pip3 install -U --user wheel mock pillow
pip3 install testresources
## above 58.3.0 you get version issues
pip3 install setuptools==58.3.0
pip3 install Cython
## install gdown to download from Google drive
pip3 install gdown
# download the wheel
gdown https://drive.google.com/uc?id=1e9FDGt2zGS5C5Pms7wzHYRb0HuupngK1
## install PyTorch 1.13.0
pip3 install torch-1.13.0a0+git7c98e70-cp38-cp38-linux_aarch64.whl

## Install torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
pip3 install -U pillow
## install gdown to download from Google drive, if not done yet
pip3 install gdown
## download TorchVision 0.13.0
gdown https://drive.google.com/uc?id=11DPKcWzLjZa5kRXRodRJ3t9md0EMydhj
## install TorchVision 0.13.0
pip3 install torchvision-0.13.0a0+da3794e-cp38-cp38-linux_aarch64.whl