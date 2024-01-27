from ..base import BaseModule


class Preprocessor(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()

    def postprocess(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()
