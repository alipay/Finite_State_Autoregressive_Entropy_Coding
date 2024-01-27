from ..base import BaseModule


class ContextModel(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    One time context extraction
    """
    def run_compress(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()

    """
    Iterable context update
    """
    def run_decompress(self, data, *args, prior=None, **kwargs):
        raise NotImplementedError()
