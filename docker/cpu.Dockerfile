# DANGER -- If you update this file, make sure to also update gpu.Dockerfile!

FROM gcr.io/kaggle-images/python:latest

# NODE

# nvm environment variables
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.16.0

# install nvm
# https://github.com/creationix/nvm#install-script
RUN curl --silent -o- https://raw.githubusercontent.com/creationix/nvm/v0.31.2/install.sh | bash

# install node and npm
RUN . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

# add node and npm to path so the commands are available
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# confirm installation
RUN node -v
RUN npm -v

# END NODE

WORKDIR /usr/src/app/kaggle_environments

# Conda boost interferes with gfootball
# RUN rm -r /opt/conda/lib/cmake/Boost-1.*
# RUN apt-get update
# RUN apt-get -y install libsdl2-gfx-dev libsdl2-ttf-dev libsdl2-image-dev
# RUN cd /tmp && \
#    git clone --single-branch --branch v2.8 https://github.com/google-research/football.git && \
#    cd football && \
#    sed -i 's/copy2/move/g' gfootball/env/observation_processor.py && \
#    sed -i 's/os\.remove/# os.remove/g' gfootball/env/observation_processor.py && \
#    pip3 install . && \
#    cd /tmp && rm -rf football

RUN pip install Flask

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments
RUN pip install .
RUN pytest

CMD kaggle-environments
