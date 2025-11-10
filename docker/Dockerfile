FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ARG MAX_JOBS 32
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHITECTURES=89;86
ENV TORCH_CUDA_ARCH_LIST="8.9;8.6" TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PIP_CACHE_DIR=/tmp/

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined requirements and common tools / libs
        autoconf automake autotools-dev build-essential ca-certificates gnupg2 \
        make cmake yasm pkg-config gdb gcovr binutils-dev libiberty-dev \
        ninja-build ccache cppcheck doxygen graphviz plantuml \
        cimg-dev \
        clang-format \
        dh-autoreconf \
        libatlas-base-dev \
        libboost-all-dev \
        liblog4cplus-dev \
        libssh2-1-dev \
        libreadline-dev \
        libcereal-dev \
        libcgal-dev \
        libblas-dev libcxsparse3 libceres-dev libeigen3-dev libfftw3-dev liblapack-dev liblapacke-dev libnlopt-dev \
        libflann-dev metis libmetis-dev libopenblas-dev libsuitesparse-dev \
        libgtk2.0-dev libglew-dev libglm-dev libglu1-mesa-dev mesa-utils \
        freeglut3-dev libassimp-dev libglfw3-dev libproj-dev libqt5opengl5-dev \
        libxrandr-dev libxxf86vm-dev \
        libgflags-dev libgoogle-glog-dev libgtest-dev \
        libfreeimage-dev libjpeg-dev libpng-dev libtiff-dev libtiff5-dev \
        libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libv4l-dev v4l-utils \
        libpcap-dev libpostproc-dev libpq-dev libswscale-dev libswresample-dev libxine2-dev \
        intel-mkl libopenmpi-dev libtbb2 libtbb-dev  \
        daemontools krb5-user libkrb5-dev \
        ibverbs-providers libibverbs1 libnuma1 libnuma-dev libpmi2-0-dev \
        libhdf5-dev protobuf-compiler libprotobuf-dev librdmacm1 libsqlite3-dev libssl-dev libtool \
        libyaml-dev libyaml-cpp-dev libicu-dev libsimpleini-dev \
        libopencv-dev libopenexr-dev \
        ffmpeg zlib1g-dev \
        ## Useful tools
        git curl wget unzip xz-utils nano vim-tiny \
        net-tools sudo htop iotop iputils-ping \
        cloc locate rsync screen tmux \
        daemontools nfs-common rsync proxychains \
        openssh-server openssh-client \
        software-properties-common \
        # Python
        python-is-python3 python3-dev python3.10-dev python3-pip python3-numpy \
        # PCL
        libpcl-dev \
        # VTK
        libvtk9-dev libvtk9.1-qt python3-vtk9 \
        # OSMesa build dependencies
        libosmesa6-dev \
        # EGL build dependencies
        libopengl-dev \
        libglvnd-dev \
        libgl-dev \
        libglx-dev \
        libegl-dev \
        # X11 utils
        mesa-utils \
        x11-apps \
        xorg-dev \
        # QT
        qtbase5-dev \
        # 
        libjsoncpp-dev \
    && apt-get purge unattended-upgrades \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###############################################################
WORKDIR /tmp

# Install Determined AI and python deps
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.3.0 &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install -U pip setuptools pathtools promise pybind11 &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

###############################################################

RUN pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 &&\
    rm -rf /tmp/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    rm -rf /tmp/*
###############################################################

WORKDIR /opt

RUN git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive &&\
    cd gaussian-splatting/ &&\
    cd submodules &&\
    cd simple-knn &&\
    python setup.py install &&\
    cd .. &&\
    cd fused-ssim &&\
    python setup.py install &&\
    cd .. &&\
    cd diff-gaussian-rasterization &&\
    python setup.py install

COPY requirement.txt /tmp/requirement.txt
RUN pip install -r /tmp/requirement.txt