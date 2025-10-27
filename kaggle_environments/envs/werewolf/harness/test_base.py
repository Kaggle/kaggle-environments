import json
import os

import litellm
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skip("Require the key to run test.")
def test_vertex_ai():
    model = "vertex_ai/deepseek-ai/deepseek-r1-0528-maas"
    file_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    with open(file_path, "r") as file:
        vertex_credentials = json.load(file)

    vertex_credentials_json = json.dumps(vertex_credentials)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        vertex_ai_project=os.environ["VERTEXAI_PROJECT"],
        vertex_ai_location=os.environ["VERTEXAI_LOCATION"],
        vertex_credentials=vertex_credentials_json,
    )
    print(response)


@pytest.mark.skip("Require the key to run test.")
def test_together():
    model = "together_ai/deepseek-ai/DeepSeek-R1"
    response = litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}])
    print(response)
