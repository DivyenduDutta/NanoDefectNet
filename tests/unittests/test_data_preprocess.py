"""

Test the data preprocessing module.
These tests dont run on CI due to lack of availability of raw dataset.
Ensure the raw dataset is available in the expected directory ie, `_RAW_DATASET_PATH`
and run tests locally.

"""

import os

from nanodefectnet.scripts.data_preprocess import read_raw_dataset

# raw dataset path
_RAW_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw",
    "LSWMD.pkl",
)


def test_read_raw_dataset():
    """
    Test the reading of the raw dataset into a pandas dataframe.
    """
    df_wafermap_raw = read_raw_dataset(_RAW_DATASET_PATH, display_meta_info=False)

    # basic existence check
    assert df_wafermap_raw is not None, "Failed to read raw dataset"
    assert (
        len(df_wafermap_raw) == 811457
    ), "Unexpected number of rows in raw dataset. Expected 811457"

    expected_columns = set(
        [
            "waferMap",
            "dieSize",
            "lotName",
            "waferIndex",
            "trainTestLabel",
            "failureType",
        ]
    )
    assert expected_columns.issubset(
        df_wafermap_raw.columns
    ), "Missing expected columns in raw dataset"
    assert (
        "trianTestLabel" not in df_wafermap_raw.columns
    ), "trianTestLabel should not be present in raw dataset"
