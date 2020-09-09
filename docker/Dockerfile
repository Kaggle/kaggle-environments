FROM gcr.io/kaggle-images/python:latest

WORKDIR /usr/src/app/kaggle_environments

# Conda boost interferes with gfootball
RUN rm -r /opt/conda/lib/cmake/Boost-1.72.0
RUN apt-get update
RUN apt-get -y install libsdl2-gfx-dev libsdl2-ttf-dev libsdl2-image-dev
RUN pip install gfootball==2.3

ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
ADD ./kaggle_environments ./kaggle_environments
RUN pip install .

CMD kaggle-environments
