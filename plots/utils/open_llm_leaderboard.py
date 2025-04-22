import pandas as pd
import re

from datasets import load_dataset
from bs4 import BeautifulSoup

def get_openllm_leaderboard_data():
    dataset = load_dataset(
        "open-llm-leaderboard/contents",
        split="train"
    ).sort("Average ⬆️", reverse=True)

    return pd.DataFrame(dataset)


def clean_text(text):

    if isinstance(text, str):
        # First remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Then remove the specific emoji tag
        text = text.replace("📑", "")
        text = text.strip()

    return text


def extract_model_info(model_path):
    # Split by '/' to separate organization and model
    parts = model_path.split('/')
    organization = parts[0]

    # Get the model part
    model_part = parts[1]

    # Extract model name (base name without version)
    model_name = re.search(r'([A-Za-z\-]+)', model_part).group(1)

    # Extract version (like 2.5, 3.3, 3.1, etc.)
    version_match = re.search(r'([0-9]+\.?[0-9]*)', model_part)
    version = version_match.group(1) if version_match else None

    # Extract parameter count (like 72B, 32B, 7B, etc.)
    param_match = re.search(r'([0-9]+(?:\.[0-9]+)?[A-Za-z]+)', model_part)
    param_count = param_match.group(1) if param_match else None

    # Extract purpose (like Instruct, Coder, Math, etc.)
    purpose_match = re.search(r'(Instruct|Coder|Math|VL|Preview|it|a[0-9]+m)(?:-[0-9]+[A-Za-z]+)?$', model_part)
    purpose = purpose_match.group(0) if purpose_match else None

    return pd.Series({
        'organization': organization,
        'model_name': model_name,
        'version': version,
        'param_count': param_count,
        'purpose': purpose
    })
