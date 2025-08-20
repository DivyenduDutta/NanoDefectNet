import yaml
from typing import Dict


def load_config(path) -> dict:
    """
    Loads and parses a YAML configuration file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_inference_config(inference_config_path: str) -> Dict:
    """
    Loads and returns the inference configuration from the specified YAML file.

    Args:
        inference_config_path (str): Path to the inference configuration YAML file.

    Returns:
        Dict: Inference configuration dictionary.
    """
    inference_config = load_config(inference_config_path)
    return inference_config
