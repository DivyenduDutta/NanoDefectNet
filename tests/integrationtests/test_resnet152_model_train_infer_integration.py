import os
import pytest
import shutil

from nanodefectnet.core.train_test_classification_model import run_train_test_model
from nanodefectnet.utils.config_utils import load_config, get_inference_config
from nanodefectnet.utils.image_utils import load_image
from nanodefectnet.core.inference_classification_model import (
    run_inference_model,
    get_trained_model_path,
)

_PATH_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_resources/configs/classifier_resnet152_aug_test.yaml",
)
cfg = load_config(_PATH_CONFIG_FILE)

_TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_resources/processed/test/Center/center_defect.png",
)

_INFERENCE_CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "test_resources/configs/inference/infer_test.yaml",
)


@pytest.fixture
def create_training_folders():
    """
    Fixture to create training folders for the model and cleanup after test is completed.
    """

    saving_dir_experiments = cfg["model"]["saving_dir_experiments"]
    saving_dir_model = cfg["model"]["saving_dir_model"]

    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # Hand control to the test
    yield

    # Cleanup training folders
    shutil.rmtree(saving_dir_experiments, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.usefixtures("create_training_folders")
def test_resnet152_train_inference() -> None:
    """
    Test the training of the ResNet152 model.
    """
    do_train = cfg["model"].get("do_train", True)
    do_test = cfg["model"].get("do_test", True)

    # 1 - train model
    run_train_test_model(cfg=cfg, do_train=do_train, do_test=do_test)

    model_name = "ResNet152"

    # 2 - rename saved model for easier loading during inference
    inference_config = get_inference_config(_INFERENCE_CONFIG_FILE_PATH)
    model_path = os.path.join(
        os.getcwd(), get_trained_model_path(model_name, inference_config)
    )
    model_dir = os.path.dirname(model_path)
    files = [
        f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
    ]
    existing_model_path = file_path = os.path.join(model_dir, files[0])
    os.rename(existing_model_path, model_path)

    image_path = _TEST_IMAGE_PATH

    image = load_image(image_path)

    # 3 - run inference
    predicted_class, predicted_score = run_inference_model(
        model_name=model_name,
        image=image,
        inference_config_file_path=_INFERENCE_CONFIG_FILE_PATH,
    )

    assert (
        predicted_score >= 0.0 and predicted_score <= 1.0
    ), "Unexpected predicted score range"
