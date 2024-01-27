from configs.class_builder import ClassBuilder, ParamSlot
import cbench.codecs

config = ClassBuilder(
    cbench.codecs.PyZstdDictCodec,
    level=ParamSlot("level", 
        choices={i: i for i in range(1, 23)},
        default=3,
    ),
    dict_size=ParamSlot("dict_size", default=32*1024)
)
