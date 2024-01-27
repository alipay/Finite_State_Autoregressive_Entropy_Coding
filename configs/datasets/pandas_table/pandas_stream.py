import cbench.data.datasets
from configs.class_builder import ClassBuilder, ParamSlot
from configs.import_utils import import_all_config_from_dir
from configs.env import DEFAULT_DATA_PATH
import os

config = ClassBuilder(
    cbench.data.datasets.PandasTableStreamDataset,
    fetch_mode=ParamSlot("fetch_mode", choices=dict(
            row='row',
            column='column',
        )
    ),
    param_group_slots=[
        ParamSlot("source", 
            choices=import_all_config_from_dir("source", caller_file=__file__, convert_to_named_param=False)
        ),
        ParamSlot("serialize", 
            choices=import_all_config_from_dir("serialize", caller_file=__file__, convert_to_named_param=False)
            # choices=dict(
            #     csv=dict(
            #         serialize_format="csv",
            #         serialize_config=dict(index=False, header=False),
            #     ),
            #     json=dict(
            #         serialize_format="json",
            #         serialize_config=dict(),
            #     ),
            # )
        ),
    ],
)
# .add_param_group_slot("source", import_config_from_module(default_source, package=__package__)) \
# .add_param_group_slot("serialize", import_config_from_module(default_serialize, package=__package__))
