# Quickstart to run werewolf and get visualization

Very quick guide for internal developers to run the kaggle werewolf code for debugging exploration
This example only uses models from vertexai for simplicity of auth

Checkout the werewolf_harness branch
```bash
git clone https://github.com/Kaggle/kaggle-environments.git
```

Set up preferred venv environment

Install the requirements for kaggle env
```bash
pip install -e kaggle-environments
```

Set up authentication for connecting to vertex
```bash
gcloud auth application-default login
gcloud config set project octo-aif-sandbox
```

Set up environment variables for auth, used in base.py
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
export VERTEXAI_PROJECT="octo-aif-sandbox"
export VERTEXAI_LOCATION="us-central1"
```

Run example program. Should be able to view out.html in a standard web browser
```bash
python kaggle_environments/envs/werewolf/test_werewolf_game.py --html out.html --json out.json --logs logs.txt --log_path log_path.txt
```