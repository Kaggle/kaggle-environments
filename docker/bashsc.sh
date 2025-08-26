# Start the orchestrator container in gVisor
# The double slash in the test directory is to force it to translate to a Linux path rather than a Windows path
docker run --runtime=runsc -it --cpus="0.8" --memory="4g" --entrypoint //bin/bash --rm --name python-simulations python-simulations-cpu