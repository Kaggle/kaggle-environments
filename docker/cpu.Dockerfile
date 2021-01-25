# DANGER -- If you update this file, make sure to also update gpu.Dockerfile!

FROM gcr.io/kaggle-images/python:latest

WORKDIR /usr/src/app/kaggle_environments

# Conda boost interferes with gfootball
RUN rm -rf /opt/conda/lib/cmake/Boost-1.72.0
RUN apt-get update
RUN apt-get -y install libsdl2-gfx-dev libsdl2-ttf-dev libsdl2-image-dev
RUN cd /tmp && \
    git clone --single-branch --branch v2.8 https://github.com/google-research/football.git && \
    cd football && \
    sed -i 's/copy2/move/g' gfootball/env/observation_processor.py && \
    sed -i 's/os\.remove/# os.remove/g' gfootball/env/observation_processor.py && \
    pip3 install . && \
    cd /tmp && rm -rf football

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments
RUN pip install .

CMD kaggle-environments
