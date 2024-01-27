"""
This script propose ClassBuilder, a general-purpose hierarchical hyper-parameter config framework for python classes.
We try to provide possibilty to split Algorithm and Experiment code.
This idea is inspired by pytorch_lightning, who manage to split model implementation (pl.LightningModule) and trainer (pl.Trainer).
New versions of pytorch_lightning has implemented a hyper-parameter system to LightningModule, which serves similar idea to this script.
However, ClassBuilder serves as a more general config framework as you can basically include any python objects into the config, \
    so that the algorithm programmer would not have to care about limit on the input.
Moreover, on the experiment programmer side, the hierarchical structure of ClassBuilder helps to manage and adjust hyper-parameter easily, \
    without having to copy-and-paste the whole parameter list (if using json or argparse).

Simple example:
Normally, hierarchical python classes are created like:
```python
pyclass1 = ClassName1(*args1, **kwargs1)
pyclass2 = ClassName2(*args2, **kwargs2)
pyclass = ClassName(pyclass1, *args, kwclass=pyclass2, **kwargs)
```
If you want to change those args and kwargs, you possibly need to recreate this classes and all classes that referece it!
With ClassBuilder, this could be done like this:
```python
pyclassbuilder1 = ClassBuilder(ClassName1, *args1, **kwargs1)
pyclassbuilder2 = ClassBuilder(ClassName2, *args2, **kwargs2)
pyclassbuilder = ClassBuilder(ClassName, pyclassbuilder1, *args, kwclass=pyclassbuilder2, **kwargs)
pyclass = pyclassbuilder.build_class()
```
Now if you want to change a param in kwargs1, you only need to call pyclassbuilder1.update_arg, and rebuild pyclassbuilder.
```python
pyclassbuilder1.update_arg(param=value)
pyclass = pyclassbuilder.build_class()
```

See other scripts in configs for more examples.

There's also a ParamSlot class that automatically register names, defaults and choices for complex parameters. \
This class is designed for compability with string-only config frameworks like json, but this is much at beta stage.

"""

from collections.abc import Callable
from collections import namedtuple
import copy
import hashlib
import json
import itertools
import inspect
from typing import Any, Hashable, Iterable, List, Sequence, Tuple, Dict, Optional, Callable, Union
# import abc

# ClassDef = namedtuple('ClassDef', ['class_or_name', 'args', 'kwargs'])


# def build_class(class_or_name, *args, **kwargs):
#     # iteratively build class in args and kwargs
#     args = [build_class(*arg) if isinstance(arg, ClassDef) else arg for arg in args]

#     if isinstance(class_or_name, Callable):
#         return class_or_name(*args, **kwargs)
#     else:
#         return eval(class_or_name)(*args, **kwargs)

# NamedParam = namedtuple('ParamDef', ['name', 'param'])
# NamedParamSlot = namedtuple('NamedParamSlot', ['name', 'default'])

def build_param_default_name(param: Any):
    if isinstance(param, NamedParamBase):
        name = param.name
    elif isinstance(param, (int, str)):
        name = "{}".format(param)
    elif isinstance(param, float):
        if abs(param) > 1e-3 and abs(param) < 1e3:
            name = "{:.4f}".format(param)
        elif param == 0:
            name = "0.0"
        else:
            name = "{:.2e}".format(param)
    elif param is None:
        name = "None"
    else:
        try:
            name = str(param)
        except:
            name = f"<Unknown type {type(param)}>"
        print(f"Warning! Param {param} does not have a default name! Represented as: {name}")
    return name

def inspect_parameters(fn):
    signature = inspect.signature(fn)
    params = signature.parameters
    return params

def inspect_default_value_dict(fn, include_no_default=False):
    params = inspect_parameters(fn)
    ret = {k:v.default for k,v in params.items() if k != 'self' and (include_no_default or v.default != inspect.Parameter.empty)}
    return ret


class NamedParamBase(object):
    @property
    def name(self):
        pass

    @property
    def param(self):
        pass

    def set_name(self, name):
        pass


class NamedParam(NamedParamBase):
    def __init__(self, name, param):
        self._name = name
        self._param = param

    @property
    def name(self) -> str:
        return self._name

    @property
    def param(self) -> Any:
        return self._param

    def set_name(self, name):
        self._name = name

class ParamChoiceKey(object):
    def __init__(self, choice: Any) -> None:
        self.choice = choice

class ParamSlot(NamedParamBase):
    def __init__(self, slot_name : str = "",
                 choices: Union[Dict[str, Any], List[Any]] = None,
                 default: NamedParamBase = None,
                 **kwargs):
        # check if the name is valid
        if '.' in slot_name:
            raise ValueError("'.' should not appear in slot name {}!".format(slot_name))
        self.slot_name = slot_name
        # self.default = default

        # choices allow dict or namedparams
        self.choices_dict = dict()
        if choices is None:
            # self.choices_dict = dict()
            pass
        elif isinstance(choices, (list, tuple)):
            for p in choices:
                if isinstance(p, NamedParamBase):
                    self.choices_dict[p.name] = p.param
                else:
                    try:
                        self.choices_dict[str(p)] = p
                    except:
                        raise ValueError("Invalid choice detected for slot {}".format(slot_name))
        elif isinstance(choices, dict):
            # self.choices_dict = dict()
            for k, p in choices.items():
                self.choices_dict[k] = p.param if isinstance(p, NamedParamBase) else p
        else:
            raise ValueError("Invalid choices for slot {}!".format(slot_name))


        if default is None and choices is not None:
            default_choice = list(self.choices_dict.keys())[0]
            default = default_choice  # NamedParam(default_choice, self.choices_dict[default_choice])
        self.default = default

        # parent
        self._parent = None

    # def set_param(self, named_param: NamedParamBase):
    #     self.named_param = named_param

    @property
    def param(self):
        # return None if self.named_param is None else self.named_param.param
        return self.default

    @property
    def name(self):
        return self.slot_name
        # param_name = "None" if self.named_param is None else self.named_param.name
        # return "{}={}".format(self.slot_name, param_name)

    def set_name(self, name: str):
        self.slot_name = name
        # if self.default is not None:
        #     self.default.set_name(name)

    @property
    def parent(self) -> NamedParamBase:
        return self._parent

    def set_parent(self, parent: NamedParamBase):
        self._parent = parent
        # if self._parent is None:
        #     self._parent = parent
        # else:
        #     print("Warning! Cannot reparent a ParamSlot")

    @property
    def choices(self) -> dict:
        return self.choices_dict

    # def is_valid_choice(self, choice):
    #     return isinstance(choice, str) and choice in self.choices_dict

    # def choose_param(self, choice):
    #     choosed_param = self.choices_dict[choice]
    #     # self.named_param = choosed_param
    #     return choosed_param

    def feed_param(self, param=None):
        # use default param?
        # if param is None:
        #     param = self.default

        # check choice value for explicit ParamChoiceKey
        if isinstance(param, ParamChoiceKey):
            param = param.choice
            assert(param in self.choices_dict)
        if isinstance(param, Hashable) and param in self.choices_dict:  # isinstance(param, str) and 
            choosed_param = self.choices_dict[param]
            print("Choosing param {} with key {} for slot '{}'"
                .format(choosed_param, param, self.name))
            return choosed_param
        else:
            # TODO: maybe check is param valid
            return param

    def __str__(self):
        # TODO: add choices?
        return "{}(name={}, default={})".format(self.__class__.__name__, self.slot_name, self.default)

# class ParamGroupSlot(ParamSlot):

class MultiLevelSlotName(object):
    def __init__(self, *args: List[str]) -> None:
        self.names = args

    def __getitem__(self, index) -> str:
        return self.names[index]

    def __len__(self):
        return len(self.names)


class ClassBuilderBase(NamedParamBase):
    def build_class(self, *args, **kwargs) -> object:
        raise NotImplementedError()

    def get_parameter(self, key : str, *args, **kwargs) -> Any:
        # raise NotImplementedError()
        return None

    def iter_parameters(self) -> Iterable[Tuple[str, Any]]:
        # raise NotImplementedError()
        pass

    def iter_slots(self) -> Iterable[Tuple[str, ParamSlot]]:
        for name, param in self.iter_parameters():
            if isinstance(param, ParamSlot):
                yield name, param

    def get_slot_data_by_name(self, slot_name: str) -> Tuple[bool, Any]:
        raise NotImplementedError()
    
    def set_slot_data_by_name(self, slot_name: str, slot_data: Any) -> bool:
        raise NotImplementedError()
    
    # Utility functions
    def get_slot_by_name(self, slot_name: str) -> Union[ParamSlot, None]:
        param = self.get_parameter(slot_name)
        if isinstance(param, ParamSlot):
            return param
        return None

    def update_slot_params(self, **kwargs) -> Any:
        # if not self.is_cloned:
        #     print("Warning! Calling update_slot_params on original Classbuilder!")
        #     print("It is recommended to call clone() before update_slot_params.")
        for name, param in kwargs.items():
            if not self.set_slot_data_by_name(name, param):
                slot = self.get_slot_by_name(name)
                if slot is not None:
                    if not isinstance(slot.parent, ClassBuilderBase):
                        raise ValueError("Slot {} parent is not properly set!".format(slot.name))
                    slot.parent.set_slot_data_by_name(slot, param)
                else:
                    raise ValueError(
                        "Slot {} not found in ClassBuilder {}! Available slots: {}"
                            .format(name, self.name, [name for name, _ in self.iter_slots()])
                    )

        # allow chaining calls
        return self    

    # def update_slot_params(self, **kwargs) -> None:
    #     for name, param in kwargs.items():
    #         slot = self.get_slot_by_name(name)
    #         # update param by slot
    #         if slot is not None:
    #             if not isinstance(slot.parent, ClassBuilderBase):
    #                 raise ValueError("Slot {} parent is not properly set!".format(slot.name))
    #             # elif slot.parent == self:
    #             #     raise ValueError("slots parent is not properly set for slot {}!".format(slot.name))
    #             slot.parent.update_param_by_slot(slot, param)
    #         else:
    #             print("Warning! Slot {} not found!".format(name))

    #     # allow chaining calls
    #     return self

    # def update_param_by_slot(self, slot: ParamSlot, param: Any) -> None:
    #     update_dict = {slot.name: param}
    #     self.update_slot_params(**update_dict)

class ClassBuilder(ClassBuilderBase):

    SUPPORTED_PARAM_TYPES = (int, float, str, list, tuple, dict, ClassBuilderBase, ParamSlot)
    
    def __init__(self, class_init: Callable, *args,
                 builder_prefix: str = None,
                 override_name: str = None,
                 param_group_slots: List[ParamSlot] = list(),
                 logger=None, # TODO: logger for classbuilder
                 **kwargs):
        """ __init__

        Args:
            class_init (Callable): Class initialize funtion (the class definition itself)
            builder_prefix (str, optional): name prefix. Defaults to None.
            override_name (str, optional): custom override name instead of hashtags. Defaults to None.
            param_group_slots (List[ParamSlot], optional): list of ParamSlot that updates arguments. Defaults to list().
        """        
        # self.class_init = class_init
        self.update_class(class_init)
        self.args = list(args)
        self.kwargs = kwargs

        # TODO: a better way to validate params? (maybe use blacklist)
        # self._validate_params(*args, **kwargs)

        # gather slots
        self._parameters = dict()
        self._slots = dict()
        self.slots_data = dict()
        for i, arg in enumerate(self.args):
            self._parameters['{}'.format(i)] = arg
            if isinstance(arg, ParamSlot):
                # add an alias for in parameters
                # self._parameters[arg.slot_name] = arg
                # set slot name if not specified
                self._setup_param_slot(arg, name=f"{i}")
        kw_defaults = inspect_default_value_dict(self.class_init)
        for kw, arg in self.kwargs.items():
            self._parameters[kw] = arg
            if isinstance(arg, ParamSlot):
                # add an alias for in parameters
                # self._parameters[arg.slot_name] = arg
                # set slot name if not specified
                # set default to class default if possible

                # NOTE: it seems some args may not have default values
                # maybe just print a warning?
                # if arg.default is None and kw not in kw_defaults:
                #     raise KeyError("Fail to set default on slot {}. Is the slot name correct?")
                
                self._setup_param_slot(arg, 
                    name=f"{kw}",
                    default=kw_defaults.get(kw),
                )
        # slots for param groups
        if isinstance(param_group_slots, Sequence):
            param_group_slots = {slot.slot_name: slot for slot in param_group_slots}
        self.param_group_slots = param_group_slots
        
        # add to cache
        for slot_name, slot in self.param_group_slots.items():
            if slot_name in self._parameters:
                raise ValueError("{} param already exists! Change to a different name for param_group_slots!".format(slot_name))
            self._setup_param_slot(slot)
            # self._parameters[slot_name] = slot
            # self._slots[slot_name] = slot

        # set parent (done in _setup_param_slot)
        # for _, slot in self._slots.items():
        #     slot.set_parent(self)

        # build slots_data
        # self.slots_data = {name: slot.param for name, slot in self._slots.items()}

        self.builder_prefix = builder_prefix if builder_prefix is not None else "undefined"
        # self.builder_name = self._build_name(prefix=builder_prefix)
        self.override_name = override_name

        # a state to give warn to param changes
        # should clone before changing params in most cases
        self.is_cloned = False

        # NOTE: just validate the given params for the given class, do not initialize a new class
        # self.build_class()

    # def _iter_slots(self):
    #     return itertools.chain(self._slots.items(), self.param_group_slots.items())

    def _setup_param_slot(self, slot: ParamSlot, name=None, default=None):
        if len(slot.slot_name) == 0:
            if name is not None:
                slot.set_name(name)
            else:
                raise NameError(f"Name for defined for slot {slot}")
        # set default to class default if possible
        if slot.default is None and default is not None:
            slot.default = default
        slot.set_parent(self)
        self._parameters[slot.slot_name] = slot
        self._slots[slot.slot_name] = slot
        self.slots_data[slot.slot_name] = slot.param

    def _validate_params(self, *args, **kwargs):
        for i, arg in enumerate(args):
            if not isinstance(arg, self.SUPPORTED_PARAM_TYPES):
                raise ValueError(
                    "Argument {} {} not supported in ClassBuilder! Consider using a ClassBuilder to wrap it!"
                    .format(i, arg))

        for kw, arg in kwargs.items():
            if not isinstance(arg, self.SUPPORTED_PARAM_TYPES):
                raise ValueError(
                    "Argument {}={} not supported in ClassBuilder! Consider using a ClassBuilder to wrap it!"
                    .format(kw, arg))

    # TODO: __copy__ or __deepcopy__?
    # this function does not copy slots data by default
    def clone(self, copy_slot_data=False):
        # TODO: maybe should call clone on ClassBuilder in args
        cloned_builder = self.__class__(
            self.class_init,
            *copy.deepcopy(self.args),
            builder_prefix=self.builder_prefix,
            override_name=self.override_name,
            param_group_slots=self.param_group_slots,
            **copy.deepcopy(self.kwargs)
        )
        cloned_builder.is_cloned = True

        if copy_slot_data:
            cloned_builder.slots_data = copy.deepcopy(self.slots_data)

        return cloned_builder

    # def __deepcopy__(self, memo):
    #     new_builder = self.clone(copy_slot_data=True)
    #     memo[id(self)] = new_builder
    #     return new_builder

    @property
    def name(self):
        return self.build_name(prefix=self.builder_prefix) \
            if self.override_name is None else self.override_name

    @property
    def param(self):
        return self

    def set_name(self, name):
        self.builder_prefix = name
        # allow chaining calls
        return self

    def set_override_name(self, name):
        self.override_name = name
        # allow chaining calls
        return self

    def get_hashtag(self, hash_length=8):
        return hashlib.sha256(self.name.encode()).hexdigest()[:hash_length]

    def get_name_under_limit(self, name_length_limit=100, hash_length=8):
        name_full = self.name
        # avoid filename too long 
        if len(name_full) > name_length_limit:
            config_hash = self.get_hashtag(hash_length=hash_length)
            trimmed_name_length = max(0, name_length_limit - len(config_hash))
            name = f"{config_hash}:{name_full[:trimmed_name_length]}..."
        else:
            name = name_full

        return name

    def build_name(self, prefix=None):
        if prefix is None:
            prefix = self.builder_prefix
        slot_defs = []
        # for slot_name, slot in self._slots.items():
        #     slot_param_name = "None"
        #     if slot_name in self.slots_data:
        #         slot_param = self.slots_data[slot_name]
        #         if slot_param is not None:
        #             slot_param_name = build_param_default_name(slot_param)
        #     slot_defs.append("{}={}".format(slot_name, slot_param_name))
        for name, param in self.iter_parameters(recursive_slot=False):
            if isinstance(param, ParamSlot): 
                slot = param
                slot_parent = slot.parent
                if slot_parent is not None:
                    # TODO: slots_data should be a function in ClassBuilderBase!
                    # slot_param_name = "None"
                    has_slot_data, slot_param = slot_parent.get_slot_data_by_name(slot.name)
                    # only append when slot_param is different from default value
                    if has_slot_data and slot_param != slot.default:
                        slot_param_name = build_param_default_name(slot_param)
                        slot_defs.append("{}={}".format(name, slot_param_name))
            else:
                continue
        return "{}({})".format(prefix, "|".join(slot_defs))

    def _build_arg(self, arg, *args, **kwargs):
        if isinstance(arg, ParamSlot):
            param_data = self.slots_data[arg.slot_name]
            param_data = arg.feed_param(param_data)
            arg = param_data
        if isinstance(arg, ClassBuilderBase):
            arg = arg.build_class(*args, **kwargs)
        # if isinstance(arg, (list, tuple)):
        #     arg = 
        return arg

    def build_class(self, *args, **kwargs):
        # iteratively build class in args and kwargs
        class_args = list()
        class_kwargs = dict()

        for arg in self.args:
            arg = self._build_arg(arg)
            class_args.append(arg)

        for kw, arg in self.kwargs.items():
            arg = self._build_arg(arg)
            class_kwargs[kw] = arg

        # update class_kwargs from param_group_slots
        for slot_name, slot in self.param_group_slots.items():
            param_data = self.slots_data[slot_name]
            param_data = slot.feed_param(param_data)
            class_kwargs.update(**param_data)

        # update class_kwargs from input kwargs
        class_kwargs.update(**kwargs)

        return self.class_init(*class_args, *args, **class_kwargs)

    def get_parameter(self, key : str, *args, **kwargs) -> Any:
        if key in self._parameters:
            return self._parameters[key]
        else:
            # TODO: could optimize!
            for name, param in self.iter_parameters():
                if name == key:
                    return param
            return None

    def iter_parameters(self, recursive=True, recursive_slot=True) -> Iterable[Tuple[str, Any]]:
        for name, param in self._parameters.items():
            yield name, param
            if recursive:
                param_tmp = param
                # convert ParamSlot to normal param for recursive slot searching
                if isinstance(param_tmp, ParamSlot) and recursive_slot:
                    param_tmp = self.slots_data.get(param_tmp.name)
                if isinstance(param_tmp, ClassBuilderBase):
                    for children_name, children_param in param_tmp.iter_parameters():
                        yield '.'.join((name, children_name)), children_param

    # def iter_slots(self) -> Iterable[Tuple[str, ParamSlot]]:
    #     for name, slot in self._slots.items():
    #         yield name, slot
    #         param = self.slots_data.get(slot.name)
    #         if isinstance(param, ClassBuilderBase):
    #             for children_name, children_slot in param.iter_slots():
    #                 yield '.'.join((name, children_name)), children_slot
    
    def iter_slot_data(self) -> Iterable[Tuple[str, Any]]:
        # for name, slot in self._slots.items():
        for name, param in self.iter_parameters():
            if isinstance(param, ParamSlot):
                assert(isinstance(param.parent, ClassBuilder))
                slot_data = param.parent.slots_data.get(param.name)
                if isinstance(slot_data, ClassBuilder):
                    yield name, slot_data.builder_prefix
                    # for children_name, children_slot_data in param.iter_slot_data():
                    #     yield '.'.join((name, children_name)), children_slot_data
                else:
                    yield name, slot_data

    # def get_slot_by_name(self, slot_name: str) -> Union[ParamSlot, None]:
    #     if '.' in slot_name:
    #         slot_name_levels = slot_name.split('.') # len>=1
    #         param = self
    #         for name_level in slot_name_levels:
    #             next_param = self.slots_data.get(name_level)
    #             if isinstance(next_param, ClassBuilderBase):
    #                 param = next_param
    #             else:
    #                 return param.get_slot_by_name(name_level)
    #     else:
    #         if slot_name in self._slots:
    #             return self._slots[slot_name]
    #         elif slot_name in self.param_group_slots:
    #             return self.param_group_slots[slot_name]
    #         else:
    #             return None
    
    def get_slot_data_by_name(self, slot_name: str) -> Tuple[bool, Any]:
        # if self owns the slot, just set the correspoding data
        if slot_name in self.slots_data:
            return True, self.slots_data[slot_name]
        else:
            slot = self.get_slot_by_name(slot_name)
            if slot is not None:
                if not isinstance(slot.parent, ClassBuilderBase):
                    raise ValueError("Slot {} parent is not properly set!".format(slot.name))
                return slot.parent.get_slot_data_by_name(slot.name)
        return False, None

    def set_slot_data_by_name(self, slot_name: str, slot_data: Any) -> bool:
        # if self owns the slot, just set the correspoding data
        if slot_name in self.slots_data:
            self.slots_data[slot_name] = slot_data
            return True
        else:
            slot = self.get_slot_by_name(slot_name)
            if slot is not None:
                if not isinstance(slot.parent, ClassBuilderBase):
                    raise ValueError("Slot {} parent is not properly set!".format(slot.name))
                return slot.parent.set_slot_data_by_name(slot.name, slot_data)
        return False

    # bottom-up update
    # def update_slot_params(self, **kwargs) -> None:
    #     if not self.is_cloned:
    #         print("Warning! Calling update_slot_params on original Classbuilder!")
    #         print("It is recommended to call clone() before update_slot_params.")
    #     for name, param in kwargs.items():
    #         if name in self.slots_data:
    #             self.slots_data[name] = param
    #         else:
    #             slot = self.get_slot_by_name(name)
    #             if slot is not None:
    #                 if not isinstance(slot.parent, ClassBuilderBase):
    #                     raise ValueError("Slot {} parent is not properly set!".format(slot.name))
    #                 elif slot.parent == self:
    #                     raise ValueError("self.slots_data is not properly set for slot {}!".format(slot.name))
    #                 slot.parent.set_slot_data_by_name(slot, param)
    #             else:
    #                 print("Warning! Slot {} not found!".format(name))

    #     # allow chaining calls
    #     return self
    
    def update_class(self, new_class : Callable, *args, clear_args=False, clear_kwargs=False, **kwargs):
        # TODO: a better instance class for class_init?
        assert(isinstance(new_class, Callable))
        self.class_init = new_class
        if clear_args:
            for idx in range(len(self.args)-1, -1, -1):
                self._remove_arg(idx)
        if clear_kwargs:
            for key in self.kwargs:
                self._remove_arg(key)

        # allow chaining calls
        return self
    
    # def update_keyword(self, old_new_dict : Dict[str, str]):
    #     for old_kw in old_new_dict:
    #         if old_kw in self.kwargs:
    #             self.kwargs[old_new_dict[old_kw]] = self.kwargs[old_kw]

    def _update_arg(self, key, arg):
        if isinstance(key, int):
            # if longer than previous args, update kwargs by order instead
            if key >= len(self.args):
                if key-len(self.args) >= len(self.kwargs.keys()):
                    raise ValueError("Cannot append args at index {}!".format(key))
                kwargs_key = list(self.kwargs.keys())[key-len(self.args)]
                print("Warning! Updating kwargs {} with extended args at index {}! This is not recommended!!".format(kwargs_key, key))
                key = kwargs_key
                self.kwargs[key] = arg
            # if i >= len(self.args):
            #     self.kwargs[kws[i-len(self.args)]] = arg
            # else:
            #     new_args[i] = arg
            else:
                self.args[key] = arg
                key = "{}".format(key)
        elif isinstance(key, str):
            self.kwargs[key] = arg

        # TODO: update inner cache
        self._parameters[key] = arg
        
        # check if there's ParamSlot
        if isinstance(arg, ParamSlot):
            # self._parameters[arg.slot_name] = arg
            # self._slots[arg.slot_name] = arg
            # self.slots_data[arg.slot_name] = arg.param
            # arg.set_parent(self)
            kw_defaults = inspect_default_value_dict(self.class_init)
            # NOTE: it seems some args may not have default values
            # maybe just print a warning?
            # if arg.default is None and kw not in kw_defaults:
            #     raise KeyError("Fail to set default on slot {}. Is the slot name correct?")
            self._setup_param_slot(arg, 
                name=key,
                default=kw_defaults.get(key),
            )

    def _remove_arg(self, key):
        if isinstance(key, int):
            # if longer than previous args, update kwargs by order instead
            if key >= len(self.args):
                if key-len(self.args) >= len(self.kwargs.keys()):
                    raise ValueError("Cannot append args at index {}!".format(key))
                kwargs_key = list(self.kwargs.keys())[key-len(self.args)]
                print("Warning! Updating kwargs {} with extended args at index {}! This is not recommended!!".format(kwargs_key, key))
                key = kwargs_key
                # self.kwargs[key] = arg
                arg = self.kwargs.pop(key)
            # if i >= len(self.args):
            #     self.kwargs[kws[i-len(self.args)]] = arg
            # else:
            #     new_args[i] = arg
            else:
                # self.args[key] = arg
                arg = self.args.pop(key)
                key = "{}".format(key)
        elif isinstance(key, str):
            arg = self.kwargs.pop(key)

        # TODO: update inner cache
        self._parameters.pop(key)
        
        # check if there's ParamSlot
        if isinstance(arg, ParamSlot):
            self._slots.pop(key)
            self.slots_data.pop(key)

    # TODO: too complicated! consider using kwargs only!
    # top-down update
    def update_args(self, *args, **kwargs):
        # if not self.is_cloned:
        #     print("Warning! Calling update_args on original ClassBuilder!")
        #     print("It is recommended to call clone() before update_args.")

        # validate params
        # self._validate_params(*args, **kwargs)

        # process args
        # new_args = list(self.args)
        # kws = list(self.kwargs.keys())
        for i, arg in enumerate(args):
            # if longer than previous args, update kwargs by order instead
            # if i >= len(self.args):
            #     self.kwargs[kws[i-len(self.args)]] = arg
            # else:
            #     new_args[i] = arg
            self._update_arg(i, arg)
        # self.args = new_args

        # update kwargs if possible
        # self.kwargs.update(**kwargs)
        for key, arg in kwargs.items():
            self._update_arg(key, arg)

        # allow chaining calls
        return self

    def remove_args(self, *args):
        for key in args:
            self._remove_arg(key)
        # allow chaining calls
        return self

    def add_all_kwargs_as_param_slot(self):
        kw_defaults = inspect_default_value_dict(self.class_init)
        for kw, arg in kw_defaults.items():
            param_slot = ParamSlot()
            # TODO: do not override existing kwargs!
            self._update_arg(kw, param_slot)
            # self._setup_param_slot(param_slot,
            #     name=kw,
            #     default=arg,
            # )
        # allow chaining calls
        return self

    def add_param_group_slot(self, slot_name: str, choices: dict, default=None):
        # if slot_name not in self.slots_data:
        if slot_name in self._parameters:
            raise ValueError("{} param already exists! Change to a different name for param_group_slots!".format(slot_name))
        slot = ParamSlot(slot_name, choices=choices, default=default)
        self._setup_param_slot(slot)
        # slot.set_parent(self)
        # # add to cache
        # self._parameters[slot_name] = slot
        # self._slots[slot_name] = slot
        # self.slots_data[slot_name] = slot.param
        self.param_group_slots[slot_name] = slot
        # allow chaining calls
        return self

    
    SLOT_ALL_CHOICES = None

    def batch_update_slot_params(self, **update_dict: Dict[str, List]) -> list:
        # get update dicts
        update_dict_all = []
        slot_names = []
        slot_params_all = []
        for slot_name, slot_params in update_dict.items():
            if slot_params is self.SLOT_ALL_CHOICES:
                slot = self.get_slot_by_name(slot_name)
                if slot is not None:
                    slot_params = slot.choices
                else:
                    raise ValueError("slot {} not found in ClassBuilder!".format(slot_name))
            if slot_params is not None:
                slot_names.append(slot_name)
                slot_params_all.append(slot_params)

        for slot_params in itertools.product(*slot_params_all):
            # print(slot_params)
            update_dict_all.append({
                name: param for name, param in zip(slot_names, slot_params)
            })

        batch_cb = []
        for update_dict_single in update_dict_all:
            cb = self.clone(copy_slot_data=True)
            cb.update_slot_params(**update_dict_single)
            batch_cb.append(cb)

        return ClassBuilderList(*batch_cb)

    # TODO: make slots definition string evaluatable
    def __str__(self):
        class_string = str(self.class_init.__name__) # TODO: some objects do not have __name__
        args_string = ", ".join([str(arg) for arg in self.args])
        kwargs_string = ", ".join(["{}={}".format(kw, str(arg)) for kw, arg in self.kwargs.items()])
        slots_string = "slots_data=({})".format(", ".join(["{}={}".format(k, v) for k, v in self.slots_data.items()]))
        return "{}({}, {})".format(
            self.__class__.__name__,
            ", ".join((class_string, args_string, kwargs_string)),
            slots_string
        )

    def to_string(self):
        return str(self)

    # NOTE: should use eval in context!
    # @staticmethod
    # def from_string(string):
    #     return eval(string)


# class ClassBuilderList(ClassBuilderBase):
#     def __init__(self, *args: List[ClassBuilder]) -> None:
#         self.class_builders = args

#         # a cache for quickly finding parameters
#         self._cached_parameters = None
#         self._cached_slots = None
    
#     def __getitem__(self, index) -> str:
#         return self.class_builders[index]

#     def __len__(self):
#         return len(self.class_builders)
    
#     # TODO: a better representative form of name for ClassBuilderList
#     @property
#     def name(self):
#         return "[{} * {}]".format(self[0].name, len(self))

#     @property
#     def param(self):
#         return self

#     def build_class(self, *args, **kwargs):
#         return [cb.build_class(*args, **kwargs) for cb in self.class_builders]

#     def get_parameter(self, key : str, *args, **kwargs) -> Any:
#         # TODO: need to check if the param has changed?
#         if self._cached_parameters is None:
#             self._cached_parameters = {k:v for k,v in self.iter_parameters()}
#         param = self._cached_parameters[key]
#         return param

#     def iter_parameters(self) -> Iterable[Tuple[str, Any]]:
#         for index, cb in enumerate(self.class_builders):
#             for name, param in cb.iter_parameters():
#                 yield '{}.'.format(index) + name, param

#     def iter_slots(self) -> Iterable[Tuple[str, ParamSlot]]:
#         for index, cb in enumerate(self.class_builders):
#             for name, slot in cb.iter_slots():
#                 yield '{}.'.format(index) + name, slot
    
#     def get_slot_by_name(self, slot_name: str) -> Union[ParamSlot, None]:
#         # TODO: need to check if the param has changed?
#         if self._cached_slots is None:
#             self._cached_slots = {k:v for k,v in self.iter_slots()}
#         slot = self._cached_slots[slot_name]
#         return slot
    

class ClassBuilderList(ClassBuilder):
    def __init__(self, *args: List[ClassBuilder]) -> None:
        self.class_builders = list(args)
        # pass ClassBuilderList to class_init will create a looped ClassBuilderList
        # after calling build_class, the result is a list of instantiated classes
        super().__init__(ClassBuilderList, *args) 
    
    def __getitem__(self, index) -> ClassBuilder:
        return self.class_builders[index]

    def __len__(self):
        return len(self.class_builders)

    # TODO: concat multiple lists?
    def __add__(self, other):
        if isinstance(other, ClassBuilderList):
            self.class_builders.extend(other.class_builders)
        elif isinstance(other, ClassBuilder):
            self.class_builders.append(other)
        else:
            raise ValueError("Cannot concat {} to ClassBuilderList".format(other))
        return self

    def build_class(self, *args, **kwargs):
        cb_list = super().build_class(*args, **kwargs)
        return cb_list.class_builders
    
    # TODO: a better representative form of name for ClassBuilderList
    @property
    def name(self):
        return "[{} * {}]".format(self.class_builders[0].name, len(self))

    # @property
    # def param(self):
    #     return self

class ClassBuilderDict(ClassBuilder):
    def __init__(self, **kwargs: Dict[str, ClassBuilder]) -> None:
        self.class_builder_dict = kwargs
        # a looped ClassBuilderDict, similar to ClassBuilderList
        super().__init__(ClassBuilderDict, **kwargs) 

    def __getitem__(self, index) -> ClassBuilder:
        return self.class_builder_dict[index]

    def build_class(self, *args, **kwargs):
        cb_dict = super().build_class(*args, **kwargs)
        return cb_dict.class_builder_dict


if __name__ == "__main__":
    class Test(object):
        def __init__(self, a, *args, b=1, other_obj=None, **kwargs):
            self.other_obj = other_obj
            self.a = a
            self.b = b

        def foo(self, c):
            if self.other_obj:
                c = self.other_obj.foo(c)
            return (self.a + c) / self.b

    cb = ClassBuilder(Test, 4, b=2, any_list=[1, 2, 3], any_dict=dict(a=2, b=3)) #, any_class=Test(0))
    test_class = cb.build_class()
    print(cb.build_class().foo(6), " should be 5")
    test_class_string = cb.to_string()
    print(test_class_string)

    cb.update_args(2, b=4)
    print(cb.build_class().foo(6), " should be 2")
    cb.update_args(2, 4)
    print(cb.build_class().foo(6), " should be 2")

    cb = eval(test_class_string)
    print(cb.build_class().foo(6), " should be 5")

    nested_cb = ClassBuilder(Test, 5, b=5, other_obj=cb)
    print(nested_cb.build_class().foo(6), " should be 2")
    nested_class_string = nested_cb.to_string()
    print(nested_class_string)

    nested_cb.update_args(7, b=4)
    print(nested_cb.build_class().foo(6), " should be 3")
    nested_cb.update_args(7, 3)
    print(nested_cb.build_class().foo(6), " should be 4")

    nested_cb = eval(nested_class_string)
    print(nested_cb.build_class().foo(6), " should be 2")

    
