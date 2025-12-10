import os
import sys
import torch
import inspect
import warnings
import functools

sys.path.append(os.getcwd())

from main.app.variables import translations

def load_model(path_or_package, strict=False):
    if isinstance(path_or_package, dict): package = path_or_package
    elif isinstance(path_or_package, (str, os.PathLike)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            package = torch.load(path_or_package, map_location="cpu", weights_only=False)
    else: raise ValueError(f"{translations['type_not_valid']} {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict: model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)

        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn(translations["del_parameter"] + key)

                del kwargs[key]

        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)

    return model

def restore_quantized_state(model, state):
    assert "meta" in state

    quantizer = state["meta"]["klass"](model, **state["meta"]["init_kwargs"])

    quantizer.restore_quantized_state(state)
    
    quantizer.detach()

def set_state(model, state, quantizer=None):
    if state.get("__quantized"):
        if quantizer is not None: quantizer.restore_quantized_state(model, state["quantized"])
        else: restore_quantized_state(model, state)
    else: model.load_state_dict(state)

    return state

def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)

        init(self, *args, **kwargs)

    return __init__