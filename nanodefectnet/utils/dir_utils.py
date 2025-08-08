import os
import shutil

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger


def clean_create_dir(path_dir: str) -> None:
    """
    Delete the existing directory and create a new one.

    Args:
        path_dir (str): The path to the directory to be cleaned and created.

    Raises:
        OSError: If the directory cannot be deleted or created.
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
