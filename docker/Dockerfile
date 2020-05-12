FROM gcr.io/kaggle-images/python

WORKDIR /usr/src/app/kaggle_environments

# gfootball dependencies
RUN apt-get update
RUN apt-get --assume-yes install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
RUN pip install gfootball

ADD ./kaggle_environments ./kaggle_environments
ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
RUN pip install .

CMD kaggle-environments