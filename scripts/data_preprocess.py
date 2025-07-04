import os
import random
import shutil
from distutils.dir_util import copy_tree

import pandas as pd

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

# dataset root path
dataset_root = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "processed"
)

path_dataset_train = os.path.join(dataset_root, "train")
path_dataset_test = os.path.join(dataset_root, "test")
path_dataset_val = os.path.join(dataset_root, "val")


def clean_create_dir(path_dir: str) -> None:
    """
    Delete the existing processed dataset directory and create a new one.

    Args:
        path_dir (str): The path to the directory to be cleaned and created.
    """

    CHECK_FOLDER = os.path.isdir(path_dir)
    if CHECK_FOLDER:
        LOGGER.info("The directory '{}' exists. Delete it".format(path_dir))
        try:
            shutil.rmtree(path_dir)
        except OSError as e:
            LOGGER.error("Error: {}".format(e.strerror))
            raise e

        CHECK_FOLDER = os.path.isdir(path_dir)
        if not CHECK_FOLDER:
            LOGGER.info("Create the directory '{}'".format(path_dir))
            os.makedirs(path_dir)
        else:
            raise OSError(f"Error: The directory '{path_dir}' was not deleted")
    else:
        LOGGER.info("Create the directory '{}'".format(path_dir))
        os.makedirs(path_dir)


if __name__ == "__main__":

    # 1 - clean and create the dataset directory
    # clean_create_dir(dataset_root)
    pass
