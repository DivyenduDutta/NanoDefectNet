import torch
import numpy as np
from typing import Tuple, Dict
from torchvision import transforms

from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.utils.constants import ModelType, ACTUAL_FAILURE_TYPE_TO_ID
from nanodefectnet.model.resnet_based import Resnet152ModelInference
from nanodefectnet.utils.config_utils import get_inference_config

LOGGER = LoggerConfig().logger


def get_trained_model_path(model_name: str, inference_config: Dict) -> str:
    """
    Get the path to the trained model for the specified model name.

    Args:
        model_name (str): Name of the model.
        inference_config (Dict): Inference configuration dictionary.

    Returns:
        str: Path to the trained model.
    """

    base_path = inference_config.get("base_path", "assets/trained_models")

    if model_name == ModelType.RESNET152.value:
        model_subpath = inference_config.get("resnet152", {}).get("model_path", "")
        return f"{base_path}/{model_subpath}"
    elif model_name == ModelType.EFFICIENTNET_B4.value:
        model_subpath = inference_config.get("efficientnet_b4", {}).get(
            "model_path", ""
        )
        return f"{base_path}/{model_subpath}"
    elif model_name == ModelType.CCT:
        model_subpath = inference_config.get("cct", {}).get("model_path", "")
        return f"{base_path}/{model_subpath}"
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_model(model_path, model_type=ModelType.RESNET152.value):
    num_classes = len(ACTUAL_FAILURE_TYPE_TO_ID)

    if model_type == ModelType.RESNET152.value:
        model = Resnet152ModelInference(num_classes)
    # else:  # efficientnet
    #     model = MushroomClassifierEfficientNet(num_classes, efficientnet_version='b4')

    # Load the model weights onto GPU
    checkpoint = torch.load(model_path, map_location=torch.device("cuda"))

    # Handle different types of saved models
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # This is a Lightning checkpoint
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix if it exists in the state dict keys
        if all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        # This is a direct state dict
        # Try to load with strict=False to ignore missing keys
        model.load_state_dict(checkpoint)

    return model


def predict(model, image, inference_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Apply transformations
    transform = get_transform(inference_config)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    image_tensor = image_tensor.to(device)

    return model.predict(image_tensor)


def get_transform(inference_config: Dict) -> transforms.Compose:
    basic_transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # because this transform pipeline expects a PIL image
            transforms.Resize(
                (inference_config["data"]["size"], inference_config["data"]["size"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(  # imagenet normalization
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return basic_transform


def run_inference_model(
    model_name: str, image: np.ndarray, inference_config_file_path: str
) -> Tuple[str, float]:
    """
    Function to run inference using the specified model.

    Args:
        model_name (str): Name of the model to use for inference.
    """
    LOGGER.info(f"Running inference with model: {model_name}")

    inference_config = get_inference_config(inference_config_file_path)

    model_path = get_trained_model_path(model_name, inference_config)
    model = load_model(model_path, model_type=model_name)
    predicted_score, predicted_class = predict(model, image, inference_config)
    return predicted_class, predicted_score
