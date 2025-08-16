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

## Simple Self Play

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