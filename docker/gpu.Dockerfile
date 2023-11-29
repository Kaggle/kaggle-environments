# DANGER -- If you update this file, make sure to also update cpu.Dockerfile!

FROM gcr.io/kaggle-gpu-images/python:latest

# NODE

# install node and npm from nodesource https://github.com/nodesource/distributions
# use a local mirror of the setup script to avoid `curl | bash`
ADD docker/nodesource_setup_14.x.sh node_setup.sh
RUN sh node_setup.sh
RUN apt-get install -y nodejs

# link the newly installed versions to /opt/node so we can prioritize these versions over the versions /opt/conda has.
RUN mkdir /opt/node && \
    ln -s /usr/bin/node /opt/node/ && \
    ln -s /usr/bin/npm /opt/node/

# add node and npm to path so the commands are available
ENV PATH /opt/node:$PATH
ENV NODE_PATH /usr/lib/node_modules

# confirm installation
RUN node -v
RUN npm -v

# END NODE

WORKDIR /usr/src/app/kaggle_environments

RUN pip install Flask

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments
RUN pip install .
RUN pytest

CMD kaggle-environments
