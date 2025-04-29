# ==================================================================
# Stage 1: Build OpenSpiel
# ==================================================================

# Use Debian 12 base for better compatibility with final gcloud image
FROM debian:12-slim as builder

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies for OpenSpiel
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    cmake \
    curl \
    git \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-tk \
    tzdata \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up a working directory for the build
WORKDIR /build_open_spiel

# --- Clone OpenSpiel Repository ---
RUN git clone --depth 1 https://github.com/deepmind/open_spiel.git .

# --- Create a fake sudo command ---
RUN echo '#!/bin/sh\nexec "$@"' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo

# Upgrade pip (using --break-system-packages for this system-level pip if needed on Debian 12)
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages

# --- Install newer CMake via pip ---
RUN pip3 install --no-cache-dir cmake --break-system-packages

# Run the OpenSpiel install script (downloads deps like Abseil)
RUN chmod +x ./install.sh && ./install.sh

# Build OpenSpiel
RUN mkdir -p build
WORKDIR /build_open_spiel/build

# Set environment variables for compilers
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

# Run CMake, explicitly passing Python executable and compilers
RUN cmake \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    ../open_spiel

# Build the code
RUN make -j$(nproc)

# --- Skip Tests ---
# RUN ctest --output-on-failure -j$(nproc)

# Reset WORKDIR for clarity before copying from this stage
WORKDIR /build_open_spiel


# ==================================================================
# Stage 2: Final image with gcloud, Kaggle Env, and OpenSpiel Runtime
# ==================================================================

FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:latest as kaggle_spiel

# Switch to root user for installations
USER root

# Install system dependencies for both Kaggle Env and OpenSpiel Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    curl \
    gpg \
    git \
    iputils-ping \
    make \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    python3-tk \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    tzdata \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# == Install Node.js (Official Nodesource Method for v20.x) ==
ENV NODE_MAJOR=20
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install nodejs -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN node -v && npm -v

# == Create and Activate Python Virtual Environment ==
RUN python3 -m venv /opt/venv
# Add venv bin directory to the PATH for subsequent RUN/CMD/ENTRYPOINT.
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install base tools WITHIN the venv
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel testresources

# == Install Kaggle Environments (using venv) ==
ARG KAGGLE_ENV_REPO=https://github.com/Kaggle/kaggle-environments.git
ARG KAGGLE_ENV_VERSION=master
ENV KAGGLE_ENV_SRC_DIR=/workspace/kaggle_environments

RUN git clone --depth 1 --branch ${KAGGLE_ENV_VERSION} ${KAGGLE_ENV_REPO} ${KAGGLE_ENV_SRC_DIR}

# Install Kaggle Environments dependencies (inside venv)
RUN python3 -m pip install --no-cache-dir "numpy<2.0"
RUN python3 -m pip install --no-cache-dir Flask vec-noise
RUN python3 -m pip install --no-cache-dir termcolor pygame

# Install Kaggle Environments in EDITABLE mode (inside venv)
WORKDIR ${KAGGLE_ENV_SRC_DIR}
RUN python3 -m pip install --no-cache-dir -e .

# == Install OpenSpiel Runtime Components (using venv) ==
ENV OPEN_SPIEL_DIR=/workspace/open_spiel

COPY --from=builder /build_open_spiel ${OPEN_SPIEL_DIR}

# Install OpenSpiel's Python runtime dependencies INTO the venv
# Ensure these are separate RUN commands
RUN python3 -m pip install --no-cache-dir -r ${OPEN_SPIEL_DIR}/requirements.txt
RUN python3 -m pip install --no-cache-dir matplotlib
# Workaround: https://github.com/google-deepmind/open_spiel/issues/1293
RUN python3 -m pip install --no-cache-dir importlib_metadata --force-reinstall

# Set PYTHONPATH to include OpenSpiel source and the compiled Python bindings
ENV PYTHONPATH=${OPEN_SPIEL_DIR}:${OPEN_SPIEL_DIR}/build/python${PYTHONPATH:+:$PYTHONPATH}

# Set final working directory to the common parent
WORKDIR /workspace

# Verify installations (optional)
RUN python3 -c "import kaggle_environments; print('Kaggle Env OK')"
RUN python3 -c "import pyspiel; print('OpenSpiel pyspiel OK')"

# The base image has its own ENTRYPOINT (gcloud).
# To get an interactive shell with the venv activated and in the workspace:
# CMD ["/bin/bash"]
