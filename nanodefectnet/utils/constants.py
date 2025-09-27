from enum import Enum

FAILURE_TYPE_TO_ID = {
    "Center": 0,
    "Donut": 1,
    "Edge-Loc": 2,
    "Edge-Ring": 3,
    "Loc": 4,
    "Random": 5,
    "Scratch": 6,
    "Near-full": 7,
    "none": 8,
}

ID_TO_FAILURE_TYPE = {v: k for k, v in FAILURE_TYPE_TO_ID.items()}

ACTUAL_FAILURE_TYPE_TO_ID = {
    "Center": 0,
    "Edge-Loc": 2,
    "Edge-Ring": 3,
    "Loc": 4,
    "Random": 5,
    "Scratch": 6,
    "Near-full": 7,
}

ACTUAL_ID_TO_FAILURE_TYPE = {v: k for k, v in ACTUAL_FAILURE_TYPE_TO_ID.items()}

TRAIN_TEST_TO_ID = {"Training": 0, "Test": 1}
ID_TO_TRAIN_TEST = {v: k for k, v in TRAIN_TEST_TO_ID.items()}


class ModelType(Enum):
    RESNET152 = "ResNet152"
    EFFICIENTNET_B0 = "EfficientNet_B0"
    EFFICIENTNET_B1 = "EfficientNet_B1"
    EFFICIENTNET_B2 = "EfficientNet_B2"
    EFFICIENTNET_B3 = "EfficientNet_B3"
    EFFICIENTNET_B4 = "EfficientNet_B4"
    CCT = "CCT"
