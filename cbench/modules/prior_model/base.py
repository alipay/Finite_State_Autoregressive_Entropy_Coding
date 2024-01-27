from typing import Any
from ..base import BaseModule


class PriorModel(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # prior extraction
    def extract(self, data, *args, **kwargs):
        raise NotImplementedError()

    # prediction using extracted prior
    def predict(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()

    """
    A quick function that combines extract and predict. May be overrided to skip latent coding for faster training.
    """
    def extract_and_predict(self, data, *args, **kwargs):
        return self.predict(data, *args, prior=self.extract(data, *args, **kwargs), **kwargs)
