from configs.class_builder import ClassBuilder, ParamSlot
import cbench.codecs

config = ClassBuilder(
    cbench.codecs.PyZlibCodec,
    compressor_config=ParamSlot("level", 
        choices={i: dict(level=i) for i in range(10)},
        default=5,
    )
)