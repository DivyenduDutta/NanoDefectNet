import os
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from nanodefectnet.utils.constants import AugmentationModelType
from nanodefectnet.utils.config_utils import load_config

from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.data.dataloader import synchronize_labels
from nanodefectnet.utils.dir_utils import clean_create_dir

LOGGER = LoggerConfig().logger


def prepare_lora_unet(
    model_id: str, mixed_precision: str, device, cfg: Dict[Any, Any]
) -> torch.nn.Module:
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "fp16" in mixed_precision else torch.float32,
    )
    pipe.to(device)

    # Get the UNet model from the pipeline
    unet = pipe.unet

    # Apply LoRA to UNet
    config = LoraConfig(
        r=cfg["model"]["lora"]["r"],  # rank of the LoRA matrices
        lora_alpha=cfg["model"]["lora"]["alpha"],
        target_modules=cfg["model"]["lora"]["target_modules"],  # attention layers
        lora_dropout=cfg["model"]["lora"]["dropout"],
        bias=cfg["model"]["lora"]["bias"],
    )
    unet = get_peft_model(unet, config)
    return unet, pipe


def get_dataloader(
    batch_size: int, train_data_dir: str, cfg: Dict[Any, Any]
) -> DataLoader:
    image_size = cfg["data"]["size"]
    basic_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(  # imagenet normalization
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Load the dataset with basic transform
    train_dataset = datasets.ImageFolder(train_data_dir, transform=basic_transform)
    synchronize_labels(train_dataset)
    defect_classes = train_dataset.classes

    wafermap_dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    return wafermap_dataloader_train, defect_classes


def fine_tune(
    num_epochs: int,
    lora_unet: torch.nn.Module,
    train_dataloader: DataLoader,
    pipe: StableDiffusionPipeline,
    defect_classes,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    noise_scheduler: DDPMScheduler,
    device,
    output_dir: str,
):

    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * num_epochs
    )

    for epoch in range(num_epochs):
        lora_unet.train()
        for step, (images, labels) in enumerate(train_dataloader):

            # move images to correct device and data type
            images = images.to(device, dtype=pipe.vae.dtype)

            # encode images to latents
            latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample()
            latents = latents.to(device=device, dtype=pipe.unet.dtype)

            # add noise
            noise = torch.randn_like(latents, device=device, dtype=latents.dtype)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Convert labels -> text prompts -> embeddings
            prompts = [defect_classes[l] for l in labels]
            tokenized = pipe.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_embeddings = pipe.text_encoder(tokenized.input_ids)[0]

            assert images.dtype == torch.float16, "Images are not in float16 precision"
            assert (
                latents.dtype == torch.float16
            ), "Latents are not in float16 precision"
            assert (
                noisy_latents.dtype == torch.float16
            ), "Noisy latents are not in float16 precision"
            assert (
                text_embeddings.dtype == torch.float16
            ), "Text embeddings are not in float16 precision"
            assert noise.dtype == torch.float16, "Noise is not in float16 precision"
            assert (
                timesteps.dtype == torch.int64
            ), "Timesteps are not in int64 precision"

            # Accumulation-aware training step
            with accelerator.accumulate(lora_unet):
                model_pred = lora_unet(
                    noisy_latents, timesteps, encoder_hidden_states=text_embeddings
                ).sample

                # Loss needs to be computed in float32 for stability
                loss = F.mse_loss(model_pred, noise.to(dtype=model_pred.dtype))

                if step == 0:
                    assert (
                        loss.dtype == torch.float32
                    ), "Loss is not in float32 precision"

                # Backward + optimizer step
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} Step {step} | Loss {loss.item():.4f}")

        # Save LoRA after each epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            lora_unet.save_pretrained(output_dir)
            accelerator.print(f"Saved LoRA weights at {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_config_file",
        type=str,
        help="Path of the LoRA training config file to use",
    )
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    LOGGER.info(f"Path of the model config file to use: {path_config_file}")

    cfg = load_config(path_config_file)

    # train dataset root path
    root_dataset_path = cfg["dataset"]["root_dataset_path"]
    path_dataset_train = cfg["dataset"]["path_dataset_train"]

    train_dataset_dir = os.path.join(root_dataset_path, path_dataset_train)

    if not os.path.isdir(train_dataset_dir):
        LOGGER.error(f"Error: The directory '{train_dataset_dir}' does not exist")
        raise OSError(f"Error: The directory '{train_dataset_dir}' does not exist")

    # Configurations for LoRA fine-tuning process
    model_id = AugmentationModelType.STABLE_DIFFUSION_V1_5.value
    batch_size = cfg["dataset"]["batch_size"]
    num_epochs = cfg["model"]["num_epoch"]
    learning_rate = cfg["model"]["lr"]
    mixed_precision = cfg["model"]["mixed_precision"]  # "no", "fp16", or "bf16"

    # 1. Setup Accelerator for distributed training
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    # 2. Add LoRA to pretrained SD model
    lora_unet, pipe = prepare_lora_unet(model_id, mixed_precision, device, cfg)

    # 3. Prepare dataloader
    dataloader, defect_classes = get_dataloader(batch_size, train_dataset_dir, cfg)

    # 4. Prepare optimizer
    optimizer = torch.optim.AdamW(lora_unet.parameters(), lr=learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 5. Prepare everything with accelerator
    lora_unet, optimizer, dataloader = accelerator.prepare(
        lora_unet, optimizer, dataloader
    )

    saving_dir_experiments = cfg["model"]["saving_dir_experiments"]
    saving_dir_model = cfg["model"]["saving_dir_model"]

    output_dir = os.path.join(saving_dir_experiments, saving_dir_model)

    clean_create_dir(output_dir)

    assert (
        lora_unet.dtype == torch.float16
    ), "LoRA UNet model is not in float16 precision"
    assert (
        next(lora_unet.parameters()).dtype == torch.float16
    ), "LoRA UNet parameters are not in float16 precision"

    fine_tune(
        num_epochs,
        lora_unet,
        dataloader,
        pipe,
        defect_classes,
        accelerator,
        optimizer,
        noise_scheduler,
        device,
        output_dir,
    )

    LOGGER.info("LoRA fine-tuning completed.")
