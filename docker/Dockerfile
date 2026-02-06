FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglfw3-dev \
    libglew-dev \
    libarmadillo-dev \
    libzmqpp-dev \
    ros-noetic-mavros \
    python3-pip \
    python3-catkin-tools \
    build-essential \
    cmake \
    unzip \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY .bak /tmp/bak

RUN unzip -q /tmp/bak/osqp.zip -d /tmp \
 && mkdir -p /tmp/osqp/build \
 && cd /tmp/osqp/build \
 && cmake -G "Unix Makefiles" .. \
 && cmake --build . \
 && cmake --build . --target install \
 && rm -rf /tmp/osqp

RUN unzip -q /tmp/bak/osqp-eigen.zip -d /tmp \
 && mkdir -p /tmp/osqp-eigen/build \
 && cd /tmp/osqp-eigen/build \
 && cmake .. \
 && make -j"$(nproc)" \
 && make install \
 && rm -rf /tmp/osqp-eigen

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir ortools

WORKDIR /work
COPY . /work

RUN source /opt/ros/noetic/setup.bash \
 && cd MARSIM_ws \
 && catkin_make

RUN source /opt/ros/noetic/setup.bash \
 && cd Air_ws \
 && catkin_make

RUN source /opt/ros/noetic/setup.bash \
 && cd Ground_ws \
 && catkin_make

ENTRYPOINT ["/bin/bash", "/work/docker/entrypoint.sh"]
CMD ["bash"]
