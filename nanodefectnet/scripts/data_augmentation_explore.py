import os
import cv2
import albumentations as A

from nanodefectnet.utils.dir_utils import clean_create_dir

from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger

IMAGE_NAME = "center_defect.png"
IMAGE_PATH = os.path.join(os.getcwd(), "assets", "test_images", IMAGE_NAME)
AUG_PATH = os.path.join(os.getcwd(), "assets", "test_images", "augmented_images")

NUM_AUG_IMAGES = 10

# List of Albumentation data augmentations - https://explore.albumentations.ai/
transform = A.Compose(
    [
        A.Flip(p=0.5),
        A.Rotate(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=45, p=0.5),
        A.Transpose(p=0.5),
    ]
)

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
name_file_no_ext = IMAGE_NAME.split(".")[0]
file_ext = IMAGE_NAME.split(".")[-1]

clean_create_dir(AUG_PATH)

LOGGER.info(f"Augmented images will be saved to: {AUG_PATH}")
LOGGER.info(f"Generating {NUM_AUG_IMAGES} augmented images from: {IMAGE_PATH}")

for i in range(NUM_AUG_IMAGES):
    augmented_image = transform(image=image)["image"]
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        os.path.join(AUG_PATH, f"{name_file_no_ext}_{i+1}.{file_ext}"), augmented_image
    )
