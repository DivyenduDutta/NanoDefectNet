import yaml


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
