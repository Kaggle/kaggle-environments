# Cloud Run Source

This directory contains files for the legacy Cloud Run function. This cloudrun function is used to serve visualizers that haven't been migrated
to use the static/vite builds that are hosted on GCS.

The cloud run function is called kaggle-simulations-v2 on the kaggle-comps-prod-eval GCP project.

The `requirements.txt` file is the source of truth for the Python dependencies of this function.
The `main.py` file is the source of truth for the Python code of this function.
