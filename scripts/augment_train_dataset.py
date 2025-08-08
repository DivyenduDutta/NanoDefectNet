import os
import shutil
import cv2
import albumentations as A

from nanodefectnet.utils.constants import FAILURE_TYPE_TO_ID

from nanodefectnet.utils.dir_utils import clean_create_dir
from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

FINAL_NUM_IMAGES_PER_CLASS = 2000

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ]
)


def make_data_augmentation(
    train_dataset_root: str, augmented_train_dataset_root: str, final_number: int
) -> None:
    """
    Perform data augmentation on the training dataset to ensure each class has a
    balanced number of images.

    Args:
        train_dataset_root (str): Path to the training dataset directory.
        augmented_train_dataset_root (str): Path to the augmented training dataset directory.
        final_number (int): The final number of images per class after augmentation.
    """

    # iterate over the classes in the train dataset directory
    sub_dirs = [
        directory
        for directory in os.listdir(train_dataset_root)
        if os.path.isdir(os.path.join(train_dataset_root, directory))
    ]
    for defect_class in sub_dirs:
        path_defect_class = os.path.join(train_dataset_root, defect_class)
        if os.path.isdir(path_defect_class):
            label = FAILURE_TYPE_TO_ID[defect_class]
            LOGGER.info("CLASS: {}  - LABEL: {}".format(defect_class, label))
            number_files = len(os.listdir(path_defect_class))
            LOGGER.info(
                f"There are {number_files} training images for the class '{defect_class}'"
            )

            path_aug_defect_class = os.path.join(
                augmented_train_dataset_root, defect_class
            )
            if not os.path.isdir(path_aug_defect_class):
                LOGGER.info("Create directory '{}'".format(path_aug_defect_class))
                os.makedirs(path_aug_defect_class)

            n_applications = round((final_number - number_files) / number_files)
            if (
                n_applications < 0
            ):  # if we already have more than the final number of augmented images
                n_applications = 0
            LOGGER.info(
                f"Number of applications for data augmentation: {n_applications}"
            )

            for train_filename in os.listdir(path_defect_class):
                path_train_image = os.path.join(path_defect_class, train_filename)
                image = cv2.imread(path_train_image)
                filename_no_ext, extension = (
                    train_filename.split(".")[0],
                    train_filename.split(".")[-1],
                )

                # copy the original image from the source dir to the dest dir (augmented dir)
                dst_file = os.path.join(path_aug_defect_class, train_filename)
                shutil.copy2(path_train_image, dst_file)

                for i in range(n_applications):
                    image = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )  # openCV reads images in BGR format
                    augmented_image = transform(image=image)["image"]
                    augmented_image = cv2.cvtColor(
                        augmented_image,
                        cv2.COLOR_RGB2BGR,  # convert back to BGR format for saving
                    )
                    new_file_name = filename_no_ext + "_" + str(i + 1) + "." + extension
                    dst_file = os.path.join(path_aug_defect_class, new_file_name)

                    cv2.imwrite(dst_file, augmented_image)

            num_images_after_aug = len(os.listdir(path_aug_defect_class))
            print(
                "Final number of files on dir '{}': {}".format(
                    path_aug_defect_class, num_images_after_aug
                )
            )
            print("-" * 60)


if __name__ == "__main__":

    # train dataset root path
    train_dataset_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed", "train"
    )

    if not os.path.isdir(train_dataset_root):
        LOGGER.error(f"Error: The directory '{train_dataset_root}' does not exist")
        raise OSError(f"Error: The directory '{train_dataset_root}' does not exist")

    # augmented train dataset root path
    augmented_train_dataset_root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "processed",
        "train_augmented",
    )

    # 1 - clean and create the dataset directory
    clean_create_dir(augmented_train_dataset_root)

    # 2 - make data augmentation
    LOGGER.info(f"Train Dataset Augmentation")
    make_data_augmentation(
        train_dataset_root, augmented_train_dataset_root, FINAL_NUM_IMAGES_PER_CLASS
    )
