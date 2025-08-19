from nanodefectnet.scripts.augment_train_dataset import _calculate_n_applications


def test_calculate_n_applications():
    """
    Test the edge cases for calculating the number of applications needed for data augmentation.
    """
    assert _calculate_n_applications(10, 50) == 3, "Should be capped at 3"
    assert _calculate_n_applications(10, 20) == 1, "Should be 1"
    assert _calculate_n_applications(10, 5) == 0, "Should be 0"
