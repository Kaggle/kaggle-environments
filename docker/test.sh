# Start the orchestrator container in Docker
docker run -it --entrypoint pytest --rm --name python-simulations python-simulations-cpu '/usr/src/app/kaggle_environments' "$@"
