from ..base import BaseModule


class Transform(BaseModule):
    """
    
    """
    def __init__(self, *args, is_invertable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_invertable = is_invertable

    def forward(self, data, *args, **kwargs):
        raise NotImplementedError()

    def inverse(self, data, *args, **kwargs):
        raise NotImplementedError()
