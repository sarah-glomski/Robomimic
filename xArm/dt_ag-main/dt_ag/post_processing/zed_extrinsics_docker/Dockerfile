# Base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,utility,video,graphics,display
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Suppress interactive prompts
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libglu1-mesa \
    git \
    vim \
    tmux \
    nano \
    htop \
    sudo \
    curl \
    wget \
    gnupg2 \
    bash-completion \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    python3-pip \
    usbutils \
    libturbojpeg0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 Humble
RUN set -x \
    && apt-get update \
    && apt-get install -y curl gnupg lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop

# Install additional ROS2 packages and development tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Install ZED SDK 4.2.5
RUN apt-get update && apt-get install -y zstd libusb-1.0-0-dev \
    && wget https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu22 -O ZED_SDK_Ubuntu22_CUDA12_v4.2.5.run \
    && chmod +x ZED_SDK_Ubuntu22_CUDA12_v4.2.5.run \
    && ./ZED_SDK_Ubuntu22_CUDA12_v4.2.5.run -- silent skip_drivers skip_tools \
    && rm -rf /var/lib/apt/lists/*

# Expose ZED SDK libs to the dynamic linker
RUN printf "/usr/local/zed/lib\n" > /etc/ld.so.conf.d/zed-sdk.conf \
    && ldconfig

# Install AprilTag via apt package
RUN apt-get update && \
    apt-get install -y ros-humble-apriltag && \
    rm -rf /var/lib/apt/lists/*

# Set up environment for ROS2 in root's bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Create workspace for AprilTag in root's home
WORKDIR /root/apriltag_ws/src
RUN git clone https://github.com/christianrauch/apriltag_ros \
    && git clone https://github.com/christianrauch/apriltag_msgs.git

# Install AprilTag library dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build the ROS2 workspace for AprilTag in root's home
WORKDIR /root/apriltag_ws

# Copy modified AprilTagNode.cpp file
COPY AprilTagNode.cpp /root/apriltag_ws/src/apriltag_ros/src/AprilTagNode.cpp
COPY tags_36h11.yaml /root/apriltag_ws/src/apriltag_ros/cfg/tags_36h11.yaml

# Note: No need to switch user, already root
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set up ZED ROS2 workspace in root's home
WORKDIR /root/zed_ros2_ws/src
RUN git clone --branch humble-v4.2.5 https://github.com/stereolabs/zed-ros2-wrapper.git

# Install ZED ROS2 wrapper dependencies
WORKDIR /root/zed_ros2_ws

# Copy modified ZED configuration file
COPY zed2i.yaml /root/zed_ros2_ws/src/zed-ros2-wrapper/zed_wrapper/config/zed2i.yaml


# Note: No need to switch user, already root
RUN apt-get update && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && rosdep install --from-paths src --ignore-src -r -y" && \
    rm -rf /var/lib/apt/lists/*

# Build the ZED ROS2 workspace in root's home
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set environment variables for both workspaces in root's bashrc
RUN echo "source /root/apriltag_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/zed_ros2_ws/install/setup.bash" >> /root/.bashrc

# Set default working directory to root's home
WORKDIR /root

# Set LD_LIBRARY_PATH for CUDA and ZED SDK
# Note: This might still be useful even for root
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/zed/lib:${LD_LIBRARY_PATH}"

# Remove CUDA compat files (no sudo needed)
RUN rm -rf /usr/local/cuda/compat

# Default command (starts a bash shell as root)
CMD ["/bin/bash"]

# Example docker run command (from your previous file)
# docker run -it \
#     --name zed \
#     --gpus all \
#     --privileged \
#     --network host \
#     --volume=/tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY=$DISPLAY \
#     -v /dev:/dev \
#     -v /dev/bus/usb:/dev/bus/usb \
#     aqiu218/zed_extrinsics:latest \
#     bash

# Zed wrapper command:
# ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i

# Apriltag pose detection command:
# ros2 run apriltag_ros apriltag_node --ros-args \
#     -r image_rect:=/zed/zed_node/left/image_rect_color \
#     -r camera_info:=/zed/zed_node/left/camera_info \
#     -p size:=0.1

# Tag in xarm base frame:
    # [0.290, -0.040, 0.010] # x, y, z

# ZED Left frame in xarm base frame:
    # [0.75, 0.25, 0.48], [0.63, 0.57, -0.35, -0.40] # x, y, z, qx, qy, qz, qw
