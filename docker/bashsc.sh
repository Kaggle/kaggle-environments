# Start the orchestrator container in gVisor
docker run --runtime=runsc -it --cpus="0.8" --memory="4g" --entrypoint //bin/bash --rm --name python-simulations python-simulations-cpu