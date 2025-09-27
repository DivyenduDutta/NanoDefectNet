import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
)
from typing import Dict, Optional, Tuple

import lightning as pl

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nanodefectnet.model.base import Model, ModelInference
from nanodefectnet.data.dataloader import create_loaders
from nanodefectnet.utils.constants import ACTUAL_ID_TO_FAILURE_TYPE, ModelType

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

#################################################################
# TRAINING VALIDATION TEST SECTION                              #
#################################################################


class EfficientNetModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        class_weights,
        unfreeze_strategy="progressive",
        model_type=ModelType.EFFICIENTNET_B0,
        dropout_rate=0.3,
    ):
        super().__init__()

        # Load the selected EfficientNet model
        if model_type == ModelType.EFFICIENTNET_B0:
            self.backbone = models.efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT
            )
            feature_dim = 1280
        elif model_type == ModelType.EFFICIENTNET_B1:
            self.backbone = models.efficientnet_b1(
                weights=EfficientNet_B1_Weights.DEFAULT
            )
            feature_dim = 1280
        elif model_type == ModelType.EFFICIENTNET_B2:
            self.backbone = models.efficientnet_b2(
                weights=EfficientNet_B2_Weights.DEFAULT
            )
            feature_dim = 1408
        elif model_type == ModelType.EFFICIENTNET_B3:
            self.backbone = models.efficientnet_b3(
                weights=EfficientNet_B3_Weights.DEFAULT
            )
            feature_dim = 1536
        elif model_type == ModelType.EFFICIENTNET_B4:
            self.backbone = models.efficientnet_b4(
                weights=EfficientNet_B4_Weights.DEFAULT
            )
            feature_dim = 1792
        else:
            raise ValueError(f"Unsupported EfficientNet version: {model_type.value}")

        # Remove the original classifier
        # The reason we dont just add our classifier here is because we want to decouple the ResNet
        # blocks from the classifier. And this is because after we get the features from the backbone,
        # we want to apply the attention mechanism before the final classification.
        self.backbone.classifier = nn.Identity()

        # Add a spatial attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Add a new classifier with dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self.class_weights = class_weights
        self.outputs: list[dict] = []
        self.train_losses: list[np.number] = []
        self.val_losses: list[np.number] = []
        self._epoch_counter = 0
        self.unfreeze_strategy = unfreeze_strategy
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Apply the initial freezing strategy
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        # First freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_strategy == "none":  # pure transfer learning
            # Keep everything frozen except the classifier
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.attention.parameters():
                param.requires_grad = True

        elif self.unfreeze_strategy == "partial":
            # Unfreeze the classifier, attention, and the last block of EfficientNet
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.attention.parameters():
                param.requires_grad = True
            # Unfreeze the last convolutional block
            for param in self.backbone.features[-1].parameters():
                param.requires_grad = True

        elif self.unfreeze_strategy == "full":
            # Unfreeze the entire model
            for param in self.backbone.parameters():
                param.requires_grad = True

        elif self.unfreeze_strategy == "progressive":
            # Start with just the classifier and attention unfrozen
            # The rest will be unfrozen progressively during training
            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.attention.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        # Progressive unfreezing based on epoch number
        if self.unfreeze_strategy == "progressive":
            # EfficientNet has 8 blocks (0-7), we'll unfreeze from the end
            if self._epoch_counter == 5:  # After 5 epochs, unfreeze the last block
                LOGGER.info(f"Unfreezing last block (epoch {self._epoch_counter})")
                for param in self.backbone.features[-1].parameters():
                    param.requires_grad = True
            elif (
                self._epoch_counter == 10
            ):  # After 10 epochs, unfreeze the second last block
                LOGGER.info(
                    f"Unfreezing second last block (epoch {self._epoch_counter})"
                )
                for param in self.backbone.features[-2].parameters():
                    param.requires_grad = True
            elif (
                self._epoch_counter == 15
            ):  # After 15 epochs, unfreeze the third last block
                LOGGER.info(
                    f"Unfreezing third last block (epoch {self._epoch_counter})"
                )
                for param in self.backbone.features[-3].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone.features(x)

        # Apply attention mechanism
        attention_map = self.attention(features)
        attended_features = features * attention_map

        # Pass through the classifier
        return self.classifier(attended_features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        start_time = time.time()
        logits = self(x)
        end_time = time.time()
        inference_time = end_time - start_time
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "inference_time_train",
            inference_time,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(
            preds, y, task="multiclass", num_classes=len(self.class_weights)
        )
        precision_val = precision_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )
        recall_val = recall_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )
        f1_val = f1_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_precision",
            precision_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall",
            recall_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_f1", f1_val, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self) -> None:
        self.val_losses.append(self.trainer.callback_metrics["val_loss"].item())
        self.log(
            "val_loss", self.val_losses[-1], on_epoch=True, prog_bar=True, logger=True
        )

    def on_train_epoch_end(self) -> None:
        self.train_losses.append(self.trainer.callback_metrics["train_loss"].item())
        self.log(
            "train_loss",
            self.train_losses[-1],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self._epoch_counter += 1

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(
            preds, y, task="multiclass", num_classes=len(self.class_weights)
        )
        precision_val = precision_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )
        recall_val = recall_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )
        f1_val = f1_score(
            preds,
            y,
            task="multiclass",
            num_classes=len(self.class_weights),
            average="macro",
        )

        self.log("test_accuracy", acc)
        self.log("test_precision", precision_val)
        self.log("test_recall", recall_val)
        self.log("test_f1", f1_val)

        return {
            "test_accuracy": acc,
            "test_precision": precision_val,
            "test_recall": recall_val,
            "test_f1": f1_val,
        }

    def configure_optimizers(self):
        # Group parameters by learning rate
        # 1. Newly added layers (classifier and attention) - higher learning rate
        new_params = list(self.classifier.parameters()) + list(
            self.attention.parameters()
        )

        # 2. Backbone layers - lower learning rate based on depth
        backbone_params = []
        if self.unfreeze_strategy in ["partial", "full", "progressive"]:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        # Create parameter groups with different learning rates
        param_groups = [
            {"params": new_params, "lr": 1e-3},
            {
                "params": backbone_params,
                "lr": 1e-5,
            },  # Much lower LR for pretrained backbone
        ]

        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        # Use cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,  # Double the restart interval after each restart
            eta_min=1e-6,  # Minimum learning rate
        )

        return [optimizer], [scheduler]


class EfficientNetModel(Model):
    def __init__(
        self,
        config: Dict,
        num_classes: int,
        load_from_ckpt: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        unfreeze_strategy: Optional[str] = "partial",
    ):
        self.config = config
        model_load_path = os.path.join(
            os.getcwd(),
            self.config["model"]["saving_dir_experiments"],
            self.config["model"]["saving_dir_model"],
        )
        if load_from_ckpt:
            # change the checkpoint name explicitly, see if there's a better way
            self.model = EfficientNetModule.load_from_checkpoint(
                os.path.join(
                    model_load_path,
                    "efficientnet-model-epoch=epoch=1-val_loss=val_loss=0.6401.ckpt",
                ),
                config=None,
            )
        else:
            self.model = EfficientNetModule(
                config["model"], num_classes, class_weights, unfreeze_strategy
            )
            checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=model_load_path,
                filename="efficientnet-model-epoch={epoch}-val_loss={val_loss:.4f}",
                save_weights_only=True,
            )
            lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(
                logging_interval="epoch"
            )
            early_stopping = pl.pytorch.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, mode="min"
            )
            self.trainer = pl.Trainer(
                max_epochs=self.config["model"]["num_epoch"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=[checkpoint_callback, lr_monitor, early_stopping],
                log_every_n_steps=10,
            )

    def train(self, cfg) -> None:
        train_dataloader, val_dataloader, _ = create_loaders(cfg)

        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        plt.figure(figsize=(10, 5))
        plt.plot(self.model.train_losses, label="Train Loss")
        plt.plot(self.model.val_losses, label="Validation Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                os.getcwd(),
                self.config["model"]["saving_dir_experiments"],
                "train_loss_plot.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def validate(self) -> None:
        _, val_dataloader, _ = create_loaders(self.config)
        self.trainer.validate(self.model, val_dataloader)

    def test(self) -> None:
        _, _, test_dataloader = create_loaders(self.config)
        self.trainer.test(self.model, test_dataloader)

    def get_params(self) -> Dict:
        return self.model.parameters()
