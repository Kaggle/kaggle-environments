FROM gcr.io/kaggle-images/python

WORKDIR /usr/src/app/kaggle_environments

RUN rm -r /opt/conda/lib/cmake/Boost-1.72.0
RUN apt-get -y install libboost-filesystem-dev libsdl2-gfx-dev libsdl2-ttf-dev libsdl2-image-dev
RUN pip3 install gfootball

ADD ./kaggle_environments ./kaggle_environments
ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
RUN pip install .

CMD kaggle-environments