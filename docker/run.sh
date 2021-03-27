# Start the orchestrator container in Docker
docker run -it --entrypoint kaggle-environments --rm -p 127.0.0.1:8080:8080/tcp --name python-simulations python-simulations-cpu "$@"