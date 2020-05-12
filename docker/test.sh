# Start the orchestrator container in Docker
# The double slash in the test directory is to force it to translate to a Linux path rather than a Windows path
docker run -it --entrypoint pytest --rm -p 127.0.0.1:8080:8080/tcp --name python-simulations python-simulations '//usr/src/app/kaggle_environments'