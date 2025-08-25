# Quickstart to run werewolf and get visualization

Very quick guide for internal developers to run the kaggle werewolf code for debugging exploration
This example only uses models from vertexai for simplicity of auth

Checkout the werewolf_harness branch
```bash
git clone https://github.com/Kaggle/kaggle-environments.git
cd kaggle-environments
git checkout werewolf_harness
```

Set up preferred venv environment

Install the requirements for kaggle env
```bash
pip install -e kaggle-environments
```

[Optional] Set up authentication for connecting to vertex
```bash
gcloud auth application-default login
gcloud config set project octo-aif-sandbox
```

Set up `.env` under project root for auth, used in base.py
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TOGETHERAI_API_KEY=...
XAI_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS="/my/path/xxx.json"
VERTEXAI_PROJECT=my-project-name
VERTEXAI_LOCATION=us-central1
GEMINI_MODEL="gemini-2.5-pro"
```

## Running a Game

The primary way to run a game is by using the `run.py` script, which uses a YAML configuration file to define all the game parameters, including the agents.

To run a game with the default configuration (`run_config.yaml`):
```bash
python kaggle_environments/envs/werewolf/scripts/run.py
```
The output, including a log file and an HTML replay, will be saved in a timestamped subdirectory inside `werewolf_run/`.

### Customizing a Run

- **Use a different configuration file:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -c path/to/your/config.yaml
  ```

- **Use random agents for a quick test:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -r
  ```

- **Enable debug mode for more verbose logging:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -d
  ```

## Running an Experiment Block

For more rigorous testing, the `run_block.py` script allows you to run a series of games in a structured block experiment. This is useful for evaluating agent performance across different role assignments and player rotations.

Each game is run as an independent process, ensuring that experiments are clean and logs are separated.

To run a block experiment with the default configuration:
```bash
python kaggle_environments/envs/werewolf/scripts/run_block.py
```
The output will be saved in `werewolf_block_experiment/`, with subdirectories for each block and game.

### Customizing an Experiment

- **Use a different configuration file:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -c path/to/your/config.yaml
  ```

- **Specify the number of blocks:**
  Each block runs a full rotation of roles for the given players.
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -b 5  # Runs 5 blocks
  ```

- **Shuffle player IDs to mitigate name bias:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -s
  ```

- **Use random agents for a quick test:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -r
  ```

### Parallel Execution

- **Run games in parallel to speed up the experiment:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -p
  ```

- **Specify the number of parallel processes:**
  If not specified, the script will calculate a reasonable default.
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -p -n 4
  ```

Note that kaggle environment by default use multiprocessing to run each agent in a separate process if debug mode is disabled. This means that the main processes you can use for each game would be greatly reduced. If you use sequential protocols e.g. round robin discussion, sequential voting, etc, we would recommend to enable debug mode `-d` to have sequential execution of each game and enable parallel processing of `run_block.py` script.

### Debugging

- **Enable debug mode to run games sequentially in the main process:**
  This is useful for stepping through code with a debugger.
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run_block.py -d
  ```

## Simple Self Play (Legacy)

Run example program. Should be able to view out.html in a standard web browser

To use random agents for quick game engine troubleshooting,
```bash
python kaggle_environments/envs/werewolf/scripts/self_play.py --use_random_agent --output_dir my/path/to/replay/dir
# or equivalently
python kaggle_environments/envs/werewolf/scripts/self_play.py -r -o my/path/to/replay
```

To use gemini for quick self-play simulation,
```bash
python kaggle_environments/envs/werewolf/scripts/self_play.py
# or if you want to use a different model and output_path versus default
python kaggle_environments/envs/werewolf/scripts/self_play.py --litellm_model_path gemini/gemini-2.5-pro --brand gemini --output_dir my/path/to/replay/dir
```

## End to End Generate Game Play and Audio
```bash
# simple testing with debug audio
python kaggle_environments/envs/werewolf/scripts/dump_audio.py -o werewolf_replay_audio --debug-audio -r -s
# full llm game play and audio
python kaggle_environments/envs/werewolf/scripts/dump_audio.py --output_dir werewolf_replay_audio --shuffle_roles
```