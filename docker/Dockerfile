FROM gcr.io/kaggle-images/python

WORKDIR /usr/src/app/kaggle_environments

ARG USE_GFOOTBALL=true
# Allow disabling gfootball as it doesn't build in Cloud Build yet.
RUN if [ "$USE_GFOOTBALL" = "true" ];\
    then \
        apt-get update && \
        apt-get --assume-yes install libsdl2-gfx-dev libsdl2-ttf-dev libboost-all-dev libsdl2-image-dev && \
        pip install gfootball; fi

ADD ./kaggle_environments ./kaggle_environments
ADD ./setup.py ./setup.py
ADD ./README.md ./README.md
ADD ./MANIFEST.in ./MANIFEST.in
RUN pip install .

CMD kaggle-environments