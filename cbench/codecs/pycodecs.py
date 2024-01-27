__all__ = [
    'PyCodec',

    # string codecs
    'PyZlibCodec',
    'PyBz2Codec',
    'PyLzmaCodec',
    'PyZstdCodec',
    'PyBrotliCodec',

    # image codecs
    # 'ImagePNGCodec',

    # trainable
    "PyZstdDictCodec",

]

import io
from .base import BaseCodec, BaseTrainableCodec

import functools
import numpy as np
from typing import List, Any

# codecs

class PyCodec(BaseCodec):
    def __init__(self, compressor, decompressor, *args, compressor_config: dict = None, decompressor_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = compressor
        self.decompressor = decompressor
        self.compressor_config = dict() if compressor_config is None else compressor_config
        self.decompressor_config = dict() if decompressor_config is None else decompressor_config

    def compress(self, data, *args, **kwargs):
        return self.compressor(data, *args, **self.compressor_config, **kwargs)

    def decompress(self, data, *args, **kwargs):
        return self.decompressor(data, *args, **self.decompressor_config, **kwargs)


# common codecs
import zlib, bz2, lzma # built-in codecs
# import zstd
import zstandard as zstd
import brotli

PyZlibCodec = functools.partial(PyCodec, zlib.compress, zlib.decompress)
PyBz2Codec = functools.partial(PyCodec, bz2.compress, bz2.decompress)
PyLzmaCodec = functools.partial(PyCodec, lzma.compress, lzma.decompress)
PyZstdCodec = functools.partial(PyCodec, zstd.compress, zstd.decompress)
PyBrotliCodec = functools.partial(PyCodec, brotli.compress, brotli.decompress)


class PyZstdDictCodec(BaseTrainableCodec):
    def __init__(self, *args, 
                 level=3,
                 dict_size=1000,
                 dict_initialize: bytes = b"",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_dict = zstd.ZstdCompressionDict(dict_initialize, *args, **kwargs)
        self.level = level
        self.dict_size = dict_size

    def compress(self, data, *args, **kwargs):
        compressor = zstd.ZstdCompressor(level=self.level, dict_data=self.compression_dict, **kwargs)
        return compressor.compress(data)

    def decompress(self, data, *args, **kwargs):
        decompressor = zstd.ZstdDecompressor(dict_data=self.compression_dict, **kwargs)
        return decompressor.decompress(data)

    def train_full(self, dataloader: List[bytes], *args, **kwargs):
        self.compression_dict = zstd.train_dictionary(
            dict_size=self.dict_size,
            samples=dataloader,
            **kwargs
        )

    def train_iter(self, data, *args, **kwargs) -> None:
        raise RuntimeError("Dictionary training does not support iterable training!")

    def get_parameters(self, *args, **kwargs) -> bytes:
        return self.compression_dict.as_bytes()

    def load_parameters(self, parameters: bytes, *args, **kwargs) -> None:
        self.compression_dict = zstd.ZstdCompressionDict(parameters, *args, **kwargs)


# image codecs
import imageio
try:
    import imageio_flif
except:
    print("imageio_flif not available! ImageFLIFCodec is disabled!")

def imageio_imwrite(data, **kwargs):
    with io.BytesIO() as bio:
        imageio.v2.imwrite(bio, data, **kwargs)
        return bio.getvalue()

def imageio_imread(data, **kwargs):
    return imageio.v2.imread(io.BytesIO(data), **kwargs)

ImageCodec = functools.partial(PyCodec, imageio_imwrite, imageio_imread)
ImagePNGCodec = functools.partial(ImageCodec, 
                                  compressor_config=dict(format="PNG"), 
                                  decompressor_config=dict(format="PNG"))
ImageWebPCodec = functools.partial(ImageCodec, 
                                   compressor_config=dict(format="WebP", lossless=True), 
                                   decompressor_config=dict(format="WebP"))
ImageFLIFCodec = functools.partial(ImageCodec, 
                                   compressor_config=dict(format="FLIF", disable_color_buckets=True), 
                                   decompressor_config=dict(format="FLIF"))

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


class PILPNGCodec(PyCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._compress
        decompressor = self._decompress
        super().__init__(compressor, decompressor, *args, **kwargs)

    def _compress(self, data, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="png", **self.compressor_config)
            return bio.getvalue()

    def _decompress(self, data, *args, **kwargs):
        return Image.open(io.BytesIO(data))


class PILWebPLosslessCodec(PyCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._compress
        decompressor = self._decompress
        super().__init__(compressor, decompressor, *args, **kwargs)

    def _compress(self, data, **kwargs):
        with io.BytesIO() as bio:
            if not isinstance(data, Image.Image):
                data = ToPILImage()(data)
            data.save(bio, format="webp", lossless=True, **self.compressor_config)
            return bio.getvalue()

    def _decompress(self, data, *args, **kwargs):
        return Image.open(io.BytesIO(data))


class PILJPEGCodec(PyCodec):
    def __init__(self, *args, **kwargs):
        compressor = self._jpeg_compress
        decompressor = self._jpeg_decompress
        super().__init__(compressor, decompressor, *args, **kwargs)

    def _jpeg_compress(self, data, quality=75, **kwargs):
        with io.BytesIO() as bio:
            ToPILImage()(data.squeeze(0)).save(bio, format="jpeg", quality=quality)
            return bio.getvalue()

    def _jpeg_decompress(self, data, *args, **kwargs):
        return ToTensor()(Image.open(io.BytesIO(data)))

