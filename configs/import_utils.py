import importlib
from modulefinder import Module
import os
import copy
import pickle
from typing import Dict, Any
from .class_builder import ClassBuilder, ParamSlot, NamedParam, NamedParamBase

# NOTE: default package is the config root
def import_config_from_module(module, package=__package__, set_module_name=True, convert_to_named_param=True):
    module_name = module.__name__
    # get relative name if package exists
    if package:
        if module_name.startswith(package):
            module_name = module_name[len(package):]
    if hasattr(module, "config"):
        config = module.config
        if convert_to_named_param and not isinstance(config, NamedParamBase):
            config = NamedParam(module_name, config)
        if set_module_name and isinstance(config, NamedParamBase):
            config.set_name(module_name)
        return copy.deepcopy(config)
    else:
        raise ValueError("Module {} does not include config variable!".format(module_name))

def import_class_builder_from_module(module, **kwargs) -> ClassBuilder:
    cb = import_config_from_module(module, **kwargs)
    assert(isinstance(cb, ClassBuilder))
    return cb

def import_raw_config_from_module(module, **kwargs):
    return import_config_from_module(module, convert_to_named_param=False)

def import_config_from_module_name(module_name, package=None, **kwargs):
    module = importlib.import_module(module_name, package=package)
    return import_config_from_module(module, package=package, **kwargs)


# TODO: package?
def import_config_from_file(filename, caller_file=None, **kwargs):
    # TODO: how to deal with config files that contains None?
    config = None
    if caller_file is not None:
        filename = os.path.join(os.path.dirname(caller_file), filename)
    if filename.endswith(".py"):
        relpath = os.path.relpath(filename, os.getcwd())
        relname = os.path.splitext(relpath)[0]
        module_name = relname.replace(os.path.sep, ".")
        config = import_config_from_module_name(module_name, **kwargs)
    elif filename.endswith(".pkl"):
        with open(filename, 'rb') as f:
            config = pickle.load(f)
    # TODO: other config formats
    return config


# TODO: recursive import?
def import_all_config_from_dir(directory, caller_file=None, convert_to_named_param=False, **kwargs) -> Dict[str, Any]:
    config_dict = dict()
    if caller_file is not None:
        directory = os.path.join(os.path.dirname(caller_file), directory)
    else:
        assert os.path.exists(directory), "Directory {} not exist! Do you forget to set caller_file?".format(directory)
    for fname in os.listdir(directory):
        config = import_config_from_file(os.path.join(directory, fname), convert_to_named_param=convert_to_named_param, **kwargs)
        if config is not None:
            config_name = os.path.splitext(fname)[0]
            config_dict[config_name] = config
    return config_dict