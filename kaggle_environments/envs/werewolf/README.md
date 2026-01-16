# Quickstart to run werewolf and get visualization

Very quick guide for internal developers to run the kaggle werewolf code for debugging exploration
This example only uses models from vertexai for simplicity of auth

Checkout the code for kaggle-environments
```bash
git clone https://github.com/Kaggle/kaggle-environments.git
cd kaggle-environments
```

Set up preferred venv environment

Install the requirements for kaggle env
```bash
pip install -e kaggle-environments
```

[Optional] For Vertex API use, set up authentication via application default credentials. Note that Google's
Gemini models can be accessed via both a consumer Developer API and enterprise Google Cloud Vertex API with more enterprise controls and features. For using Developer API authentication the GEMINI_API_KEY environment variable should be set and for Vertex AI there are a number of methods with most common being gcloud authentication to your project as below. See [documentation](https://ai.google.dev/gemini-api/docs/migrate-to-cloud) for more deatils
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Set up `.env` under project root for authentication for, used in base.py, for models that will be used.
```
# Note - Gemini can be accessed via Developer API or via Google Cloud Vertex API for enterprise feature support.
# Google Developer API will need GEMNI_API_KEY set for authentication 
# Cloud Vertex API most common access with gcloud auth login above, but other alternatives available
# Developer API 
GEMINI_API_KEY=..
# Vertex API
GOOGLE_APPLICATION_CREDENTIALS="/my/path/xxx.json" # Optional if different from default location
VERTEXAI_PROJECT=MY_PROJECT_ID # name of your project
VERTEXAI_LOCATION=LOCATION # e.g. us-central1

# See individual APIs for authentication details
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
TOGETHERAI_API_KEY=...
XAI_API_KEY=...
```

## Running a Game

The primary way to run a game is by using the `run.py` script, which uses a YAML configuration file to define all the game parameters, including the agents.

To run a game with the default configuration (`run_config.yaml`). Note that this will use the Google Developer API for Gemini and require setting GEMINI_API_KEY in environment. See [Developer API Documentation](https://ai.google.dev/gemini-api/docs/api-key) for details.
```bash
python kaggle_environments/envs/werewolf/scripts/run.py
```
The output, including a log file and an HTML replay, will be saved in a timestamped subdirectory inside `werewolf_run/`.

### Customizing a Run

- **Use a different configuration file:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -c path/to/your/config.yaml
  ```

- **Example configuration file using Vertex API:**
Note that for authentication with Vertex the application default credentials need to be set with gcloud command (see above) or alternative service account, api key, etc. See [Vertex documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys) for details. In the config file the Vertex AI API will be specified with 'llm/vertex_ai/gemini-2.5-flash' instead of 'llm/gemini/gemini-2.5-flash' for Developer API. 
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py \
    --config_path kaggle_environments/envs/werewolf/scripts/configs/run/vertex_api_example_config.yaml \
    --output_dir output_dir
  ```
- **Use random agents for a quick test:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -r
  ```

- **Enable debug mode for more verbose logging:**
  ```bash
  python kaggle_environments/envs/werewolf/scripts/run.py -d
  ```

### Configuring Agents
Each agent's configuration looks like the following
```yaml
    - role: "Villager"
      id: "gemini-2.5-pro"
      thumbnail: "https://storage.googleapis.com/kaggle-static/game-arena/werewolf/thumbnails/gemini.png"
      agent_id: "llm/gemini/gemini-2.5-pro"
      display_name: "gemini/gemini-2.5-pro"
      agent_harness_name: "llm_harness"
      chat_mode: "text"
      enable_bid_reasoning: false
      llms:
        - model_name: "gemini/gemini-2.5-pro"
```
- `id`: is the unique id of the agent. In the werewolf game, the player will be uniquely 
refereed to by the moderator as this id as well as all natural language and structured text logs.
It can be a human name like "Alex" or the model's name or any unique string.
- `thumbnail`: a thumbnail url that will be rendered by the `html_renderer` as avatar for the agent.
- `agent_id`: this is the agent identifier used by kaggle environment to initialize an agent instance, e.g. `"random"` for random agent.
We prepared LLM based harness compatible with `litellm` library. You can use `"llm/<litellm_model_name>"` to specify the LLM you want e.g. `"llm/gemini/gemini-2.5-pro"`.
The supported LLMs can be found at `kaggle_environments/envs/werewolf/werewolf.py`.
- `display_name`: this is a name you want to show in the player card that's visible only in the html rendered by `html_renderer`.
If left blank there will be no separate display name shown. This is used primarily to disambiguate id and the underlying model, e.g. id -> `Alex (gemini-2.5-pro)` <- display_name. Not used in game logic.
- `agent_harness_name`: a placeholder for you to record the agent harness name. Not used in game logic.
- `chat_mode`: This only impact instruction sets for the agent harness. 
If set to `audio`, a different instruction will be given to the LLM agent to generate audio friendly messages.
- `enable_bid_reasoning`: only useful for `BidDrivenDiscussion` protocol. If enabled, the LLM agents will use reasoning for all bid actions.
- `llms`: This is only for recording the models used in the harness. It's an array to support multi LLM setup in the future.

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

## Viewing Replays

To view a text-based summary of a replay JSON (useful for analyzing LLM reasoning):
```bash
python kaggle_environments/envs/werewolf/scripts/print_werewolf_llm.py path/to/replay.json
```

For better readability with colors, pipe to `less`:
```bash
# --force-color ensures colors are preserved even when piping
python kaggle_environments/envs/werewolf/scripts/print_werewolf_llm.py path/to/replay.json --force-color | less -R
```

## Running Evaluation
To run the evaluation scripts located in `kaggle_environments/envs/werewolf/eval/`, you'll need to install several additional dependencies. These are used for data manipulation, progress tracking, plotting, and advanced metrics calculation.

### Evaluation Dependencies
The evaluation scripts require specific Python packages. These are strictly enforced; the scripts will fail with an error if they are missing.

**For Game Loading (`loaders.py`):**
- `pandas`: For structured data manipulation.
- `tqdm`: For progress tracking.

**For Metrics & Visualization (`metrics.py`):**
- All `loaders.py` dependencies.
- `plotly`: For interactive plots.
- `kaleido`: for static image export (optional but recommended).
- `openskill`: For TrueSkill/OpenSkill ratings.
- `polarix`: For Game Theoretic Evaluation (GTE).

**Install all dependencies:**
```bash
uv pip install pandas tqdm plotly kaleido openskill polarix
```
For linux,
```bash
plotly_get_chrome
sudo apt update && sudo apt-get install libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2
```

### Scripts Overview

#### 1. Game Record Extraction (`loaders.py`)
Extracts game data from massive amounts of JSON replay files into a structured CSV format. This script is designed to be memory-efficient, using parallel processing to handle thousands of replays without consuming excessive RAM.

**Key Features:**
- **Memory Efficient:** Processes games in parallel workers and extracts only necessary fields.
- **Cost Analysis:** Extracts token usage (prompt/completion) and API costs per agent.
- **Robustness:** Handles shuffled player IDs and logs errors to a separate file.
- **Progress Tracking:** Displays a progress bar during extraction.

**Usage:**
```bash
python kaggle_environments/envs/werewolf/eval/loaders.py <input_dir> <output_path> [options]
```

**Options:**
- `input_dir`: Directory containing `.json` replay files (recursive search).
- `output_path`: Path to save the extracted CSV file.
- `--log-file`: (Optional) Path to save error logs for failed extractions.
- `--max-workers`: (Optional) Number of parallel worker processes.

**Example:**
```bash
python kaggle_environments/envs/werewolf/eval/loaders.py ./werewolf_run ./results.csv --log-file extraction_errors.log
```

#### 2. Metrics & Evaluation (`metrics.py`)
Calculates comprehensive performance metrics for agents, including Win Rates, OpenSkill ratings, and Game Theoretic Evaluation (GTE) scores. It also generates visualization plots for the Pareto Frontier of performance vs. cost.

**Key Features:**
- **Game Theoretic Evaluation (GTE):** Computes Nash equilibrium-based ratings using `polarix`.
- **Pareto Frontier:** Visualizes the trade-off between Agent Cost ($) and Performance (GTE Rating).
- **Metric Breakdown:** Analyzes Win Rate, Survival Rate (KSR), and specialized Werewolf metrics (IRP, VSS).
- **Error Handling:** Configurable error logging for game loading issues.

**Usage:**
```bash
python kaggle_environments/envs/werewolf/eval/metrics.py <input_dir> [options]
```

**Options:**
- `input_dir`: Directory containing `.json` replay files.
- `--error-log`: (Optional) Path to log game loading errors (default: `game_loading_errors.log`).
- `--gte-tasks`: (Optional) Comma-separated list of tasks for GTE (default: `win_dependent`).
- `--output-prefix`: (Optional) Prefix for generated plot files (e.g., `experiment_1_`).

**Example:**
```bash
python kaggle_environments/envs/werewolf/eval/metrics.py ./werewolf_run --output-prefix my_experiment_
```
