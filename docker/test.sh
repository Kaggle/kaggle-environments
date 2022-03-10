# Start the orchestrator container in Docker
# The double slash in the test directory is to force it to translate to a Linux path rather than a Windows path
docker run -it --entrypoint pytest --rm --name python-simulations python-simulations-cpu '/usr/src/app/kaggle_environments' "$@"
