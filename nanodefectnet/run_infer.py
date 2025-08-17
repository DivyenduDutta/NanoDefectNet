import argparse

from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.core.inference_classification_model import run_inference_model
from nanodefectnet.utils.image_utils import load_image

LOGGER = LoggerConfig().logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to use")
    parser.add_argument("--image", type=str, help="Path to the image file")
    opt = parser.parse_args()

    # 1 - get model name
    model_name = opt.model_name
    LOGGER.info(f"Name of the model to use: {model_name}")

    image_path = opt.image
    LOGGER.info(f"Classifying wafermap image at: {image_path}")

    image = load_image(image_path)

    # 2 - run inference
    predicted_class, predicted_score = run_inference_model(
        model_name=model_name, image=image
    )
    LOGGER.info(f"Predicted class: {predicted_class}, Score: {predicted_score}")
