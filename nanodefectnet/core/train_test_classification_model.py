import os
from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.utils.reproducibility import seed_everything

from nanodefectnet.model.resnet_based import Resnet152Model
from nanodefectnet.data.dataloader import get_class_weights

# import mlflow

LOGGER = LoggerConfig().logger


def run_train_test_model(cfg, do_train, do_test):

    seed_everything(42)
    # mlflow.set_experiment(cfg["model"]["model_name"])

    # 2 - Get class weights
    class_weights = get_class_weights(cfg)

    # 3 - Initialize the model
    if cfg["model"]["model_name"] == "resnet152":
        model = Resnet152Model(
            cfg,
            num_classes=cfg["model"]["num_classes"],
            load_from_ckpt=False,
            class_weights=class_weights,
            unfreeze_strategy=cfg["model"]["unfreeze_strategy"],
        )
    else:
        raise ValueError(f"Invalid model type {cfg['model']['model_name']} provided")

    # 4 - Train and test the model
    if do_train:
        LOGGER.info("Training model...")
        model.train(cfg)
    if do_test:
        LOGGER.info("Testing model...")
        model.test()
