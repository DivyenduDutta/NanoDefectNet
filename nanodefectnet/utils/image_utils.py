import cv2
import numpy as np


def load_image(image_path: str):
    # load the image_byte
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_from_bytes(image_bytes: bytes):
    # Convert raw bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # reads as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    return image
