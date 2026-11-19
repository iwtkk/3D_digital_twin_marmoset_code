# Docker image that reproduces scripts/00_setup/install.sh
#
#   git submodule update --init          # mmdetection / mmpose / marmo3Dpose must be checked out
#   docker build -t marmoset-dt .
#   (with Blender for Step 3: add --build-arg INSTALL_BLENDER=1)
#   (rootless podman without /etc/subuid: add --build-arg ROOTLESS_NO_SUBUID=1)
#   docker run --gpus all -it --rm -v $PWD:/workspace -v /path/to/data:/data marmoset-dt
#
# Notes
#  - nvidia/cuda 11.6 has no Ubuntu 22.04 image, so Ubuntu 20.04 is used with a standalone
#    Python 3.10.12 installed by uv (deadsnakes no longer provides python3.10 for Ubuntu 20.04).
#  - mmcv-full 1.6.2 has no prebuilt wheel for Python 3.10 (the cu113/torch1.10 index only has cp36-cp39),
#    so it is compiled from source against torch 1.13.1+cu116, exactly as install.sh does on Python 3.10.
#    This step takes a long time (tens of minutes). Set TORCH_CUDA_ARCH_LIST to your GPU to shorten it.

FROM docker.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Workaround for rootless podman without /etc/subuid (only when --build-arg ROOTLESS_NO_SUBUID=1):
# the container cannot use any UID/GID other than 0, so apt's _apt sandbox user and
# chown/chgrp calls in package post-install scripts fail. During the apt step, run apt as root
# and make chown/chgrp ignore errors; the real binaries are restored right after.
# Has no effect with normal Docker (default ROOTLESS_NO_SUBUID=0).
ARG ROOTLESS_NO_SUBUID=0
RUN if [ "$ROOTLESS_NO_SUBUID" = "1" ]; then \
        echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-no-sandbox && \
        for c in chown chgrp; do \
            dpkg-divert --local --rename --add /bin/$c && \
            printf '#!/bin/sh\n/bin/%s.distrib "$@" 2>/dev/null || true\n' $c > /bin/$c && chmod 755 /bin/$c; \
        done; \
    fi

# System packages (the X11/EGL libraries are needed by the optional Blender below)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential ninja-build \
        ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 tzdata \
        xz-utils libx11-6 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libegl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ROOTLESS_NO_SUBUID" = "1" ]; then \
        for c in chown chgrp; do rm -f /bin/$c && dpkg-divert --local --rename --remove /bin/$c; done; \
        rm -f /etc/apt/apt.conf.d/99-no-sandbox; \
    fi

# Python 3.10.12 + venv. Ubuntu 20.04 has no python3.10 package (deadsnakes no longer serves focal),
# so a standalone CPython build is installed with uv.
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv
RUN uv python install 3.10.12 && uv venv --seed --python 3.10.12 $VIRTUAL_ENV
# The standalone build reports clang as its compiler; build C/C++/CUDA extensions with gcc instead.
ENV CC=gcc CXX=g++
ENV PATH=$VIRTUAL_ENV/bin:$PATH
RUN pip install --upgrade "pip<24.1" "setuptools<70" wheel

WORKDIR /workspace

# ---- [1/6] PyTorch + torchvision -----------------------------------------
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu116

# ---- [2/6] mmcv-full (built from source, CUDA ops enabled) ----------------
# No GPU is visible during `docker build`, so the CUDA build must be forced.
ARG TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda
RUN pip install mmcv-full==1.6.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

RUN pip install numpy==1.22.4
# lap 0.4.0 must be compiled against the installed numpy 1.22.4; with build isolation pip would
# build it against numpy 2.x and fail.
RUN SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-build-isolation lap==0.4.0

# ---- [3/6] mmdetection / [4/6] mmpose (editable, from submodules) ---------
COPY scripts/02_train/mmdetection scripts/02_train/mmdetection
RUN pip install -e scripts/02_train/mmdetection
COPY scripts/02_train/mmpose scripts/02_train/mmpose
RUN pip install -e scripts/02_train/mmpose

# ---- [5/6] requirements.txt ----------------------------------------------
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# cv2.omnidir (3D keypoint alignment -> NPZ) is only in the contrib build; same as install.sh.
RUN pip install --no-deps opencv-contrib-python==4.6.0.66
# The repository is bind-mounted with the host user's UID; let git work on it as root.
RUN git config --global --add safe.directory '*'

# ---- [6/6] marmo3Dpose requirements (commented out in install.sh) --------
# RUN pip install -r scripts/03_run_marmo3dpose/marmo3Dpose/requirements.txt

# ---- [+] PyYAML for Blender -> third_party/ ------------------------------
# Installed outside /workspace so a bind mount of the repo does not hide it;
# symlinked into place below.
RUN pip install --target /opt/third_party PyYAML

# ---- [+] Blender (optional, for Step 3: animate / render) ---------------
# docker build --build-arg INSTALL_BLENDER=1 [--build-arg BLENDER_VERSION=5.2.2] ...
# Official Linux build from download.blender.org, installed in /opt/blender and linked to
# /usr/local/bin/blender. Cycles renders on the GPU with `docker run --gpus all`
# (NVIDIA_DRIVER_CAPABILITIES must include compute; set below).
ARG INSTALL_BLENDER=0
ARG BLENDER_VERSION=5.2.2
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
RUN if [ "$INSTALL_BLENDER" = "1" ]; then \
        v="$BLENDER_VERSION" && mm="${v%.*}" \
        && curl -fsSL "https://download.blender.org/release/Blender${mm}/blender-${v}-linux-x64.tar.xz" \
             | tar -xJ -C /opt \
        && mv "/opt/blender-${v}-linux-x64" /opt/blender \
        && ln -s /opt/blender/blender /usr/local/bin/blender \
        && blender --version | head -n 1; \
    fi

# ---- Rest of the repository ----------------------------------------------
COPY . .
RUN [ -e third_party ] || ln -s /opt/third_party third_party

CMD ["/bin/bash"]
