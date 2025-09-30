import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

from nanodefectnet.utils.constants import (
    ACTUAL_FAILURE_TYPE_TO_ID,
    AugmentationModelType,
)

from nanodefectnet.utils.dir_utils import clean_create_dir
from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

FINAL_NUM_IMAGES_PER_CLASS = 2000


def generate_augmented_images(
    model_id: str, augmented_train_dataset_root: str, wafer_lora: str, final_number: int
) -> None:

    # Load the fine-tuned LoRA model into the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, wafer_lora)

    pipe.unet = pipe.unet.merge_and_unload()

    for failure_type in ACTUAL_FAILURE_TYPE_TO_ID.keys():
        LOGGER.info(f"Generating images for class: {failure_type}")
        prompt = f"{failure_type} wafermap defect"
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            num_images_per_prompt=final_number,
        ).images

        os.makedirs(
            os.path.join(augmented_train_dataset_root, failure_type), exist_ok=True
        )

        for i, img in enumerate(images):
            img.save(
                os.path.join(augmented_train_dataset_root, failure_type, f"gen_{i}.png")
            )


if __name__ == "__main__":

    # LoRA fine-tuned SD model path
    wafer_lora = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "checkpoints",
        "lora_finetuned_sd",
        "models",
    )

    if not os.path.isdir(wafer_lora):
        LOGGER.error(f"Error: The directory '{wafer_lora}' does not exist")
        raise OSError(f"Error: The directory '{wafer_lora}' does not exist")

    # GenAI augmented train dataset root path
    augmented_train_dataset_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "processed",
        "train_augmented_genai",
    )

    # 1 - clean and create the augmented dataset directory
    clean_create_dir(augmented_train_dataset_root)

    model_id = AugmentationModelType.STABLE_DIFFUSION_V1_5.value

    # 2 - make data augmentation
    LOGGER.info(f"Train Dataset Augmentation using GenAI - Started...")
    generate_augmented_images(
        model_id, augmented_train_dataset_root, wafer_lora, FINAL_NUM_IMAGES_PER_CLASS
    )
    LOGGER.info(f"Train Dataset Augmentation using GenAI - Completed")
