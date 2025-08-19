import cv2
import numpy as np
import pytest

from nanodefectnet.scripts.augment_train_dataset import make_data_augmentation


@pytest.fixture
def dummy_image(tmp_path):
    """
    Create a dummy dataset with one class and a couple of fake images.
    """
    class_dir = tmp_path / "train" / "Scratch"
    class_dir.mkdir(parents=True)

    # make 2 dummy images (black 64x64 squares)
    for i in range(2):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / f"img_{i}.jpg"), img)

    return tmp_path


def test_make_data_augmentation_runs(dummy_image):
    """
    Test the data augmentation function.
    """
    train_root = dummy_image / "train"
    augmented_root = dummy_image / "train_augmented"

    # C:\\Users\\<user_name>\\AppData\\Local\\Temp\\pytest-of-dutta\\pytest-0\\test_make_data_augmentation_ru0\\
    # Pytest keeps around the last 3 test runs
    # print(str(train_root))

    # Run augmentation function
    make_data_augmentation(
        train_dataset_root=str(train_root),
        augmented_train_dataset_root=str(augmented_root),
        final_number=10,  # ask for more images than we have
    )

    # Check augmented directory exists
    aug_class_dir = augmented_root / "Scratch"
    assert aug_class_dir.exists(), "Augmented class dir was not created"

    files = list(aug_class_dir.iterdir())
    assert len(files) >= 2, "At least the original images should be copied"
    assert any("_1" in f.name for f in files), "Expected at least one augmented image"

    # Spot check: images are valid (cv2 can read them)
    img = cv2.imread(str(files[0]))
    assert img is not None and img.shape == (64, 64, 3)
