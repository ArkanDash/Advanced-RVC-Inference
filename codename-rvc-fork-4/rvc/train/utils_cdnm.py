import os
import glob
import torch
import numpy as np
import soundfile as sf
from collections import OrderedDict
import matplotlib.pyplot as plt

MATPLOTLIB_FLAG = False

from itertools import chain

def verify_optimizer_has_all_params(models, optimizer, model_names=("mpd","cqt")):
    # Collect the real Parameter objects the optimizer knows about:
    opt_params = {
        p
        for g in optimizer.param_groups
        for p in g["params"]
        if p.requires_grad
    }

    missing_total = 0
    for model, name in zip(models, model_names):
        model_ps = [p for p in model.parameters() if p.requires_grad]
        missing = [p for p in model_ps if p not in opt_params]

        print(f"[VERIFY] Model `{name}`:")
        print(f"  • trainable params:  {len(model_ps)}")
        print(f"  • in optimizer:      {len(model_ps) - len(missing)}")
        if missing:
            print(f"  • X MISSING:          {len(missing)}")
            missing_total += len(missing)
        else:
            print("  • V All present")

    if missing_total:
        raise RuntimeError(
            f"Optimizer is missing {missing_total} trainable parameters!"
        )
    print("V Optimizer covers all parameters.")




def check_optimizer_coverage_old(models, optimizer):
    for model_idx, model in enumerate(models):
        param_map = {id(p): name for name, p in model.named_parameters()}
        missing_params = []

        for p in model.parameters():
            if id(p) not in optimizer.state:
                if p.requires_grad:
                    name = param_map.get(id(p), "unknown")
                    missing_params.append(name)

        model_name = ["mpd", "cqt"][model_idx]
        if missing_params:
            print(f"[WARNING] X Not all parameters of {model_name} restored in optimizer!")
            print(f" Missing ({len(missing_params)}): {missing_params}")
        else:
            print(f"[OK] All parameters of {model_name} restored in optimizer.")



def check_optimizer_coverage(models, optimizer, model_names=("mpd","cqt")):
    # Grab the real Parameter objects the optimizer knows about
    opt_params = {
        p
        for g in optimizer.param_groups
        for p in g["params"]
        if p.requires_grad
    }

    for model, name in zip(models, model_names):
        model_ps = [p for p in model.parameters() if p.requires_grad]
        missing = [p for p in model_ps if p not in opt_params]
        if missing:
            print(f"[WARNING] X {len(missing)} params of {name} not hooked into optimizer!")
        else:
            print(f"[OK] All parameters of {name} are in the optimizer.")



def save_checkpoints_old(models, optimizer, learning_rate, iteration, checkpoint_path):
    """
    Save multiple models and a shared optimizer state to a checkpoint file.

    Args:
        models (list[torch.nn.Module]): List of models to save.
        optimizer (torch.optim.Optimizer): Shared optimizer.
        learning_rate (float): Current learning rate.
        iteration (int): Current iteration or epoch.
        checkpoint_path (str): Destination path for the checkpoint.
    """
    model_names = ["mpd", "cqt"]
    assert len(models) == len(model_names), "Expected exactly two models: mpd and cqt"

    checkpoint_data = {
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }

    for model, name in zip(models, model_names):
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        checkpoint_data[name] = state_dict

    # Backwards compatibility: remap key names
    checkpoint_data = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"),
        ".parametrizations.weight.original0", ".weight_g",
    )

    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved model '{checkpoint_path}' (epoch {iteration})")

