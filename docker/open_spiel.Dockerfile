# NOTE -- this follows cpu.Dockerfile, similar changes are needed to run gpu.Dockerfile for GPU support

FROM gcr.io/kaggle-images/python:latest

# NODE

# install node and npm from nodesource https://github.com/nodesource/distributions
# use a local mirror of the setup script to avoid `curl | bash`
ADD docker/nodesource_setup_14.x.sh node_setup.sh
RUN apt-get update && \
    sh node_setup.sh && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# link the newly installed versions to /opt/node so we can prioritize these versions over the versions /opt/conda has.
RUN mkdir /opt/node && \
    ln -s /usr/bin/node /opt/node/node && \
    ln -s /usr/bin/npm /opt/node/npm

# add node and npm to path so the commands are available
ENV PATH /opt/node:$PATH
ENV NODE_PATH /usr/lib/node_modules

# confirm installation
RUN node -v && npm -v

# END NODE

WORKDIR /usr/src/app/kaggle_environments

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments

# install kaggle-environments
RUN pip install Flask bitsandbytes accelerate vec-noise jax gymnax==0.0.8 && pip install . && pytest

# begin OpenSpiel integration

# 1. Install OpenSpiel Build Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang \
    # cmake will be installed via pip for a potentially newer version
    curl \
    git \
    make \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install a newer CMake via pip, as recommended for OpenSpiel.
RUN pip install --no-cache-dir cmake

# 2. Clone and Prepare OpenSpiel Source Code
WORKDIR /opt
RUN git clone --depth 1 https://github.com/deepmind/open_spiel.git
WORKDIR /opt/open_spiel

# Run OpenSpiel's install.sh script.
RUN chmod +x ./install.sh && ./install.sh

# 3. Build OpenSpiel from Source
RUN mkdir -p build
WORKDIR /opt/open_spiel/build
# Define environment variables for the C and C++ compilers to be used for OpenSpiel.
ENV CC_FOR_OPENSPIEL_BUILD=/usr/bin/clang
ENV CXX_FOR_OPENSPIEL_BUILD=/usr/bin/clang++
# Configure the OpenSpiel build using CMake.
# CMake will use /opt/open_spiel/open_spiel as the source directory.
RUN cmake \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_C_COMPILER=${CC_FOR_OPENSPIEL_BUILD} \
    -DCMAKE_CXX_COMPILER=${CXX_FOR_OPENSPIEL_BUILD} \
    -DCMAKE_BUILD_TYPE=Release \
    ../open_spiel

# Compile OpenSpiel. Uses all available processor cores for faster compilation.
RUN make -j$(nproc)

# 4. Configure Python Environment for OpenSpiel
# Add OpenSpiel's compiled Python bindings and source directory to PYTHONPATH.
ENV PYTHONPATH=/opt/open_spiel:/opt/open_spiel/build/python${PYTHONPATH:+:$PYTHONPATH}

# Install OpenSpiel's Python runtime dependencies.
RUN pip install --no-cache-dir -r /opt/open_spiel/requirements.txt

# Install additional Python packages.
RUN pip install --no-cache-dir matplotlib

# Apply a workaround for a known importlib_metadata issue with OpenSpiel.
RUN pip install --no-cache-dir importlib_metadata --force-reinstall

# 5. Tests
# Use a neutral directory for performing import tests.
WORKDIR /tmp
# Double check kaggle-environments
RUN python3 -c "import kaggle_environments; print('Kaggle Env OK')"
# Test if OpenSpiel's `pyspiel` module can be imported.
RUN python3 -c "import pyspiel; print('OpenSpiel (pyspiel) imported successfully.')"
# Verify JAX.
RUN python3 -c "import jax; import jax.numpy as jnp; print(f'JAX version: {jax.__version__} imported successfully.')"


# Revert to the original working directory from the user's Dockerfile.
WORKDIR /usr/src/app/kaggle_environments
CMD kaggle-environments
