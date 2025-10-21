# Just build the orchestrator container
path=$(dirname $0)
# cd to the parent directory to include kaggle_environments folder in Docker build context
cd $path/..
docker build -f ./docker/Dockerfile --target gpu --build-arg BASE_IMAGE=gcr.io/kaggle-gpu-images/python:latest -t python-simulations-gpu .
cd -
