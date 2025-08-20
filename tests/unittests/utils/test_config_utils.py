import os
import pytest

from nanodefectnet.utils.config_utils import load_config

_CONFIG_FILE_PATH = os.path.join(os.getcwd(), "configs/classifier_resnet152_aug.yaml")


@pytest.mark.unittest
@pytest.mark.runonci
def test_load_config() -> None:
    """
    Test the yaml config file loading utility function.
    """
    config = load_config(_CONFIG_FILE_PATH)
    assert config is not None, "Failed to load config"
    assert "model" in config, "Config is missing 'model' key"
    assert "model_name" in config["model"], "Config is missing 'model_name' key"
    assert "resnet152" == config["model"]["model_name"], "Unexpected model name"
