from configs.class_builder import ClassBuilder, ParamSlot
import cbench.codecs

config = ClassBuilder(
    cbench.codecs.PyZstdCodec,
    compressor_config=ParamSlot("level", 
        choices={i: dict(level=i) for i in range(1, 23)},
        default=3,
    )
)
