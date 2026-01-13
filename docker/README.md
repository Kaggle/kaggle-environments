# Docker Image

This image encapsulates the kaggle-environments library, its dependencies, and agent execution environment.  
This image is hosted at `gcr.io/kaggle-images/python-simulations`

## Usage

- `./build.sh` will build the image including any local changes to kaggle_environments.
- `./run.sh` will pass any arguments to the kaggle-environments command line tool running in docker. Note that this also binds port 8080 to run `kaggle-environments http-server` commands.
  - `./run.sh list`
  - `./run.sh run --environment connectx --agent random random`
  - `./run.sh http-server`

## Example for how to render a replay into a local html file

- `[python3 -m kaggle_environments.main|run.sh] load --in path/to/test-episode.json --environment open_spiel_chess --display html --out test-replay.html`
