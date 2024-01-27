from configs.class_builder import ClassBuilder, ParamSlot
import cbench.codecs

config = ClassBuilder(
    cbench.codecs.PyBrotliCodec,
    compressor_config=ParamSlot("level", 
        choices={i: dict(quality=i) for i in range(12)},
        default=dict(quality=11),
    )
)
