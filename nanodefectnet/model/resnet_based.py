import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from torchvision.models import ResNet152_Weights
from typing import Dict, Optional, Tuple

import lightning as pl

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nanodefectnet.model.base import Model, ModelInference
from nanodefectnet.data.dataloader import create_loaders
from nanodefectnet.utils.constants import ACTUAL_ID_TO_FAILURE_TYPE

#################################################################
# TRAINING VALIDATION TEST SECTION                              #
#################################################################


class Resnet152Module(pl.LightningModule):
    def __init__(
        self, config: Dict, num_classes: int, class_weights, unfreeze_strategy="partial"
    ):
        super().__init__()
        self.model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.config = config
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.class_weights = class_weights
        self.outputs: list[Dict] = []
        self.train_losses: list[np.Number] = []
        self.val_losses: list[np.Number] = []
        self._epoch_counter = 0
        self.unfreeze_strategy = unfreeze_strategy

        # Apply the initial freezing strategy
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        # First freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        if self.unfreeze_strategy == "none":
            # Keep everything frozen except the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif self.unfreeze_strategy == "partial":
            # Unfreeze the last layer and the last block of ResNet
            for param in self.model.fc.parameters():
                param.requires_grad = True
            for param in self.model.layer4.parameters():  # Last ResNet block
                param.requires_grad = True

        elif self.unfreeze_strategy == "full":
            # Unfreeze the entire model
            for param in self.model.parameters():
                param.requires_grad = True

        elif self.unfreeze_strategy == "progressive":
            # Start with just the last layer unfrozen
            # The rest will be unfrozen progressively during training
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        # Progressive unfreezing based on epoch number
        if self.unfreeze_strategy == "progressive":
            if self._epoch_counter == self.config["epoch_start_unfreeze_layer4"]:
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
            elif self._epoch_counter == self.config["epoch_start_unfreeze_layer3"]:
                for param in self.model.layer3.parameters():
                    param.requires_grad = True
            elif self._epoch_counter == self.config["epoch_start_unfreeze_layer2"]:
                for param in self.model.layer2.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        start_time = time.time()
        logits = self(x)
        end_time = time.time()
        inference_time = end_time - start_time
        self.log(
            "inference_time_train",
            inference_time,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        return loss

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
        self.outputs.append({"preds": preds, "y": y})
        return {"preds": preds, "y": y}

    def on_test_epoch_end(self):
        preds = torch.cat([out["preds"] for out in self.outputs])
        y = torch.cat([out["y"] for out in self.outputs])
        accuracy = accuracy_score(y.cpu(), preds.cpu())
        precision = precision_score(y.cpu(), preds.cpu(), average="macro")
        recall = recall_score(y.cpu(), preds.cpu(), average="macro")
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_precision", precision, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_recall", recall, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        start_time = time.time()
        logits = self(x)
        end_time = time.time()
        inference_time = end_time - start_time
        self.log(
            "inference_time",
            inference_time,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_losses.append(self.trainer.callback_metrics["val_loss"].item())
        self.log(
            "val_loss", self.val_losses[-1], on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        # Use different learning rates for different parts of the network
        # This is especially important when using progressive unfreezing

        # Group parameters by layers for different learning rates
        if (
            self.unfreeze_strategy == "progressive"
            or self.unfreeze_strategy == "partial"
        ):
            # Higher learning rate for newly added layers, lower for pretrained layers
            params = [
                {
                    "params": self.model.fc.parameters(),
                    "lr": self.config["lr"]["fcn"],
                },  # Higher LR for the new classification layer
            ]

            # Add unfrozen backbone layers with lower learning rate
            if any(p.requires_grad for p in self.model.layer4.parameters()):
                params.append(
                    {
                        "params": self.model.layer4.parameters(),
                        "lr": self.config["lr"]["layer4"],
                    }
                )
            if any(p.requires_grad for p in self.model.layer3.parameters()):
                params.append(
                    {
                        "params": self.model.layer3.parameters(),
                        "lr": self.config["lr"]["layer3"],
                    }
                )
            if any(p.requires_grad for p in self.model.layer2.parameters()):
                params.append(
                    {
                        "params": self.model.layer2.parameters(),
                        "lr": self.config["lr"]["layer2"],
                    }
                )
            if any(p.requires_grad for p in self.model.layer1.parameters()):
                params.append(
                    {
                        "params": self.model.layer1.parameters(),
                        "lr": self.config["lr"]["layer1"],
                    }
                )

            optimizer = torch.optim.AdamW(
                params, weight_decay=self.config["weight_decay"]
            )
        else:
            # Simple optimizer for full or none strategies
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["lr"]["overall"],
                weight_decay=self.config["weight_decay"],
            )

        # Use cosine annealing scheduler with warm restarts for better convergence
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config["scheduler"]["T_0"],  # Restart every 5 epochs
                T_mult=self.config["scheduler"][
                    "T_mult"
                ],  # Double the restart interval after each restart
                eta_min=self.config["scheduler"]["eta_min"],  # Minimum learning rate
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "cosine_lr",
        }

        return [optimizer], [lr_scheduler]


class Resnet152Model(Model):
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
            self.model = Resnet152Module.load_from_checkpoint(
                os.path.join(
                    model_load_path,
                    "resnet152-model-epoch=epoch=1-val_loss=val_loss=0.6401.ckpt",
                ),
                config=None,
            )
        else:
            self.model = Resnet152Module(
                config["model"], num_classes, class_weights, unfreeze_strategy
            )
            checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                dirpath=model_load_path,
                filename="resnet152-model-epoch={epoch}-val_loss={val_loss:.4f}",
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


#################################################################
# INFERENCE SECTION                                             #
#################################################################


class Resnet152ModuleInference(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet152(
            weights=None
        )  # because for inference we use weights from the trained model
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class Resnet152ModelInference(ModelInference):
    def __init__(self, num_classes: int):
        self.model = Resnet152ModuleInference(num_classes)

    def load_state_dict(self, state_dict: Dict):
        self.model.load_state_dict(state_dict, strict=False)

    def predict(self, input_image) -> Tuple[float, str]:
        self.model.eval()
        device = next(self.model.parameters()).device
        input_image = input_image.to(device)
        if input_image.ndim == 3:  # no batch dimension
            input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            logit = self.model(input_image)
            preds = torch.nn.Softmax(dim=1)(logit)
            pred_values, pred_labels = torch.max(preds, 1)
        predicted_score = round(pred_values.cpu().numpy().tolist()[0], 4)
        predicted_class = ACTUAL_ID_TO_FAILURE_TYPE[
            pred_labels.cpu().numpy().tolist()[0]
        ]
        return predicted_score, predicted_class
