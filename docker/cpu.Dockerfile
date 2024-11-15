# DANGER -- If you update this file, make sure to also update gpu.Dockerfile!

FROM gcr.io/kaggle-images/python:latest

# NODE

# install node and npm from nodesource https://github.com/nodesource/distributions
# use a local mirror of the setup script to avoid `curl | bash`
ADD docker/nodesource_setup_14.x.sh node_setup.sh
RUN sh node_setup.sh && apt-get install -y nodejs

# link the newly installed versions to /opt/node so we can prioritize these versions over the versions /opt/conda has.
RUN mkdir /opt/node && \
    ln -s /usr/bin/node /opt/node/ && \
    ln -s /usr/bin/npm /opt/node/

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

# SET UP KAGGLE-ENVIRONMENTS CHESS
# minimal package to reduce memory footprint
RUN mkdir ./kaggle_environments_chess
RUN cp -r ./kaggle_environments/* ./kaggle_environments_chess/
RUN rm -rf ./kaggle_environments
# remove other runtimes
RUN find ./kaggle_environments_chess/envs -mindepth 1 -maxdepth 1 ! -name "chess" -type d -exec rm -rf {} +
# pyclean
RUN rm -rf ./kaggle_environments_chess/__pycache__; rm -rf ./kaggle_environments_chess/envs/__pycache__; rm -rf ./kaggle_environments_chess/envs/chess/__pycache__; true
RUN find ./kaggle_environments_chess/ -name "*.pyc" -exec rm -f {} \;

# rename pip package
RUN sed -i 's/kaggle-environments/kaggle-environments-chess/g' ./setup.py
RUN sed -i 's/kaggle_environments/kaggle_environments_chess/g' ./setup.py
RUN sed -i 's/kaggle_environments/kaggle_environments_chess/g' ./MANIFEST.in

# install kaggle-environments-chess
RUN pip install . && pytest

CMD kaggle-environments
