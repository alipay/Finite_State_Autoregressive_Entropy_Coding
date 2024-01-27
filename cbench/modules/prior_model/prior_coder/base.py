from typing import Any
from ..base import BaseModule

# TODO: PriorCoder has the same interface with EntropyCoder! Consider merge them!
class PriorCoder(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, input, *args, **kwargs) -> bytes:
        raise NotImplementedError()

    def decode(self, byte_string, *args, **kwargs) -> Any:
        raise NotImplementedError()
