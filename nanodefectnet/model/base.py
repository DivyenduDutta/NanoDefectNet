from abc import ABC
from abc import abstractmethod
from typing import Dict

import numpy as np


class Model(ABC):
    @abstractmethod
    def train(self, cfg) -> None:
        """
        Performs training of model. The exact train implementations are model specific.
        :param train_datapoints: List of train datapoints
        :param val_datapoints: List of validation datapoints that can be used
        :param cache_featurizer: Whether or not to cache the model featurizer
        :return:
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass

    @abstractmethod
    def predict(self, input_image) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return the model-specific parameters such as number of hidden-units in the case
        of a neural network or number of trees for a random forest
        :return: Dictionary containing the model parameters
        """
        pass
