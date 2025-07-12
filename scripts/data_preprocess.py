import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

from nanodefectnet.utils.constants import (
    FAILURE_TYPE_TO_ID,
    ID_TO_FAILURE_TYPE,
    TRAIN_TEST_TO_ID,
)
from nanodefectnet.utils.dir_utils import clean_create_dir


def read_raw_dataset(
    path_raw_dataset: str, display_meta_info: Optional[bool] = False
) -> pd.DataFrame:
    """
    Read the raw dataset from the specified path and return a DataFrame.

    Args:
        path_raw_dataset (str): The path to the raw dataset file.
        display_meta_info (bool): If True, display metadata information about the dataset. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the wafer map data.
    """
    df_wafermap = pd.read_pickle(path_raw_dataset)

    # Rename the column 'trianTestLabel' to 'trainTestLabel'
    df_wafermap.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    if display_meta_info:
        LOGGER.info("Raw dataset loaded successfully")
        LOGGER.debug(f"Number of samples in raw dataset: {len(df_wafermap)}")
        LOGGER.debug("-" * 50)
        LOGGER.debug(f"Sample data:\n{df_wafermap.head()}")
        LOGGER.debug("-" * 50)
        LOGGER.debug(f"Sample info:\n")
        df_wafermap.info()  # cant really log this, it will be printed to console though
        LOGGER.debug("-" * 50)
        LOGGER.debug(f"Sample description:\n{df_wafermap.describe()}")

    return df_wafermap


def remove_data_with_missing_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    One hot encodes the 'failureType' and 'trainTestLabel' columns in the DataFrame,
    and removes rows where these columns contain missing information.

    Args:
        df (pd.DataFrame): The DataFrame containing raw wafer map data.

    Returns:
        pd.DataFrame: A DataFrame with missing data removed.
    """

    total_wafers = df.shape[0]
    LOGGER.info(
        f"Number of elements in dataset before removing missing values: {total_wafers}"
    )

    df["failureNum"] = df.failureType.apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else -1
    )
    df["trainTestNum"] = df.trainTestLabel.apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else -1
    )

    df = df.replace(
        {"failureNum": FAILURE_TYPE_TO_ID, "trainTestNum": TRAIN_TEST_TO_ID}
    )
    df = df.astype({"failureNum": "int", "trainTestNum": "int"})

    # We remove only the rows with missing labels for column 'failureNum' and we dont have to
    # do this for 'trainTestNum'. Because interestingly, the number of datapoints with missing train/test
    # label == the number of datapoints with missing failure label and they're are the same rows.
    # The number of datapoints in this is much less than if we included failureType=None but we
    # dont because failureType=None is not something we"re interested in.
    # More detailed analysis in the EDA notebook.
    df_withpattern = df[(df["failureNum"] >= 0) & (df["failureNum"] <= 7)]
    df_withpattern = df_withpattern.reset_index()
    LOGGER.info(
        f"Number of elements in dataset after removing missing values: {df_withpattern.shape[0]}"
    )

    return df_withpattern


def add_wafermap_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column 'waferMapDim' to the DataFrame that contains the dimensions of the wafer map.
    The dimensions are represented as a tuple (rows, columns).

    Args:
        df (pd.DataFrame): The DataFrame containing wafer map data.

    Returns:
        pd.DataFrame: The DataFrame with the new 'waferMapDim' column added
    """

    df["waferMapDim"] = df.waferMap.apply(lambda x: (x.shape[0], x.shape[1]))
    return df


def filter_by_wafermap_dimension(
    df: pd.DataFrame, min_dim: int, max_dim: int
) -> pd.DataFrame:
    """
    Filter the DataFrame by wafer map dimensions.

    Args:
        df (pd.DataFrame): The DataFrame containing wafer map data.
        min_dim (int): The minimum wafer map dimension to keep.
        max_dim (int): The maximum wafer map dimension to keep.

    Returns:
        pd.DataFrame: A DataFrame filtered by wafer map dimensions.
    """

    if "waferMapDim" not in df.columns:
        LOGGER.info("Adding wafer map dimensions to the DataFrame")
        df = add_wafermap_dimension(df)
    df_filtered = df[df["waferMapDim"] == (min_dim, max_dim)]
    LOGGER.info(
        f"Number of elements in dataset after filtering by wafer map dimension: {df_filtered.shape[0]}"
    )
    return df_filtered


def convert_pixel_classes_to_channels(
    df: pd.DataFrame, min_dim: int, max_dim: int
) -> pd.DataFrame:
    """
    Converts 2D wafer map pixel class labels into multi-channel binary format.
    Originally, each wafermap has either 0,1 or 2 in each pixel. We are simply extracting
    these classes into three separate channels in a 3D array format.

    Each pixel in the original 'waferMap' array is a class label:
        0 = background
        1 = non-defect foreground
        2 = defect foreground

    This function transforms each 2D waferMap into a 3D array (H, W, 3),
    where each channel is a binary mask for one of the three classes.

    Args:
        df (pd.DataFrame): DataFrame with a 'waferMap' column containing 2D arrays of class labels.
        min_dim (int): Expected height of each waferMap.
        max_dim (int): Expected width of each waferMap.

    Returns:
        pd.DataFrame: DataFrame with the 'waferMap' column replaced by (H, W, 3) arrays.

    Raises:
        ValueError: If any waferMap has unexpected dimensions.
    """

    LOGGER.info("One-hot encoding the wafer map pixels...")

    # Stack all wafer maps into a 3D array (N, H, W)
    wafer_maps = np.stack(df["waferMap"].values)

    # Optional sanity check
    if wafer_maps.shape[1:] != (min_dim, max_dim):
        raise ValueError(
            f"Expected wafer maps of shape {(min_dim, max_dim)}, got {wafer_maps.shape[1:]}"
        )

    # One-hot encode (assumes 3 pixel classes: 0, 1, 2)
    one_hot_maps = np.eye(3)[
        wafer_maps.astype(int)
    ]  # shape: (N, H, W, 3) with each pixel value as 0 or 1

    # Replace 'waferMap' column with encoded data
    df["waferMap"] = list(one_hot_maps)

    LOGGER.debug(f"New shape per waferMap: {df['waferMap'].iloc[0].shape}")
    return df


def save_wafer_images(df: pd.DataFrame, split_name: str, output_dir: str) -> None:
    """
    Saves waferMaps from a DataFrame as images into subfolders based on their failure type.

    Args:
        df (pd.DataFrame): DataFrame containing wafer maps.
        split_name (str): One of 'train', 'val', 'test' to designate the folder.
        output_dir (str): Root directory where images will be saved.
    """
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    for _, row in df.iterrows():
        img = (row["waferMap"] * 255).astype(np.uint8)  # Scale to 0-255 just in case
        lot = str(row["lotName"])
        wafer = str(row["waferIndex"])

        failure_type = ID_TO_FAILURE_TYPE[row["failureNum"]]
        subfolder = os.path.join(output_dir, split_name, failure_type)
        os.makedirs(subfolder, exist_ok=True)

        filename = f"{lot}_{wafer.split('.')[0]}.png"
        filepath = os.path.join(subfolder, filename)

        Image.fromarray(img).save(filepath)

    LOGGER.info(
        f"Saved {len(df)} images of {split_name} dataset to '{os.path.join(output_dir, split_name)}'"
    )


def make_train_val_test_split(
    df: pd.DataFrame, root_processed_dataset_path: str
) -> None:
    """
    Splits the DataFrame into train, validation, and test sets, and saves the images.

    Args:
        df (pd.DataFrame): The DataFrame containing wafer map data.
        root_processed_dataset_path (str): The root directory where processed dataset images will be saved.
    """
    # labels = df["trainTestNum"]
    # LOGGER.debug(f"The percentage of test data as per the original labels is : {labels.value_counts()[1] / len(labels) * 100:.2f}%")

    # 80-20 train-test split
    LOGGER.info("Making 80-20 train-test split from the full dataset...")
    total_data_length = int(0.8 * len(df))
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_test = df_shuffled[:total_data_length], df_shuffled[total_data_length:]
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # 70-10 train val split
    LOGGER.info("Making 70-10 train-val split from the train dataset...")
    train_data_length = int(0.7 * len(df_train))
    df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_val = (
        df_train_shuffled[:train_data_length],
        df_train_shuffled[train_data_length:],
    )
    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    LOGGER.info(f"Train data percentage: {len(df_train) / len(df) * 100:.2f}%")
    LOGGER.info(f"Validation data percentage: {len(df_val) / len(df) * 100:.2f}%")
    LOGGER.info(f"Test data percentage: {len(df_test) / len(df) * 100:.2f}%")

    save_wafer_images(df_train, "train", root_processed_dataset_path)
    save_wafer_images(df_val, "val", root_processed_dataset_path)
    save_wafer_images(df_test, "test", root_processed_dataset_path)


if __name__ == "__main__":

    # raw dataset path
    raw_dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw", "LSWMD.pkl"
    )

    # processed dataset root path
    processed_dataset_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed"
    )

    # 1 - clean and create the dataset directory
    clean_create_dir(processed_dataset_root)

    # 2 - read the raw dataset
    df_wafermap_raw = read_raw_dataset(raw_dataset_path, display_meta_info=True)

    # 3 - remove data with missing labels
    df_wafermap_cleaned = remove_data_with_missing_labels(df_wafermap_raw)

    # 4 - filter by wafer map dimension keeping only the most common wafer map dimension ie, (25, 27)
    # See the EDA notebook for more details on this
    # TODO: Do this for now and later we can think about resizing/standardizing dimensions
    df_wafermap_filtered = filter_by_wafermap_dimension(
        df_wafermap_cleaned, min_dim=25, max_dim=27
    )

    # 5 - one-hot encode the wafer map pixels
    df_wafermap = convert_pixel_classes_to_channels(
        df_wafermap_filtered, min_dim=25, max_dim=27
    )

    # 6 - make train-val-test split and save the images
    make_train_val_test_split(df_wafermap, processed_dataset_root)
