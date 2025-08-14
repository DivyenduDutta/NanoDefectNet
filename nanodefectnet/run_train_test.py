import argparse
import os
from shutil import copy

from nanodefectnet.utils.logger import LoggerConfig
from nanodefectnet.utils.config_utils import load_config
from nanodefectnet.src.train_test_classification_model import run_train_test_model

LOGGER = LoggerConfig().logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_config_file", type=str, help="Path of the model config file to use"
    )
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    LOGGER.info(f"Path of the model config file to use: {path_config_file}")
    cfg = load_config(path_config_file)

    saving_dir_experiments = cfg["model"]["saving_dir_experiments"]
    saving_dir_model = cfg["model"]["saving_dir_model"]

    # 1 -  create the directories with the structure required by the project
    LOGGER.info("Create the project structure")
    LOGGER.info("Experiments saved in location : %s", saving_dir_experiments)
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    LOGGER.info(f"Trained models saved in {saving_dir_model}")
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    experiment_save_path = os.path.join(os.getcwd(), saving_dir_experiments)
    LOGGER.info(f"Copying config file to {experiment_save_path}")
    copy(path_config_file, experiment_save_path)

    # 2 - run train and test
    do_train = cfg["model"].get("do_train", True)
    do_test = cfg["model"].get("do_test", True)
    run_train_test_model(cfg=cfg, do_train=do_train, do_test=do_test)
