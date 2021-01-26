# DANGER -- If you update this file, make sure to also update cpu.Dockerfile!

FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /usr/src/app/kaggle_environments

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments
RUN pip install .

CMD kaggle-environments
