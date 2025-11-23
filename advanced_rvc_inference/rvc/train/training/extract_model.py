import os
import sys
import torch
import hashlib
import datetime

from collections import OrderedDict

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, config
from main.inference.training.utils import replace_keys_in_dict

def extract_model(ckpt, sr, pitch_guidance, name, model_path, epoch, step, version, hps, model_author, vocoder, energy_use):
    try:
        logger.info(translations["savemodel"].format(model_dir=model_path, epoch=epoch, step=step))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        opt = OrderedDict(weight={key: (value if not config.device.startswith("privateuseone") else value.detach().cpu()).to(torch.float16 if config.is_half else torch.float32) for key, value in ckpt.items() if "enc_q" not in key})
        opt["config"] = [hps.data.filter_length // 2 + 1, 32, hps.model.inter_channels, hps.model.hidden_channels, hps.model.filter_channels, hps.model.n_heads, hps.model.n_layers, hps.model.kernel_size, hps.model.p_dropout, hps.model.resblock, hps.model.resblock_kernel_sizes, hps.model.resblock_dilation_sizes, hps.model.upsample_rates, hps.model.upsample_initial_channel, hps.model.upsample_kernel_sizes, hps.model.spk_embed_dim, hps.model.gin_channels, hps.data.sample_rate]
        opt["epoch"] = f"{epoch}epoch"
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = int(pitch_guidance)
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()
        opt["model_hash"] = hashlib.sha256(f"{str(ckpt)} {epoch} {step} {datetime.datetime.now().isoformat()}".encode()).hexdigest()
        opt["model_name"] = name
        opt["author"] = model_author
        opt["vocoder"] = vocoder
        opt["energy"] = energy_use

        torch.save(replace_keys_in_dict(replace_keys_in_dict(opt, ".parametrizations.weight.original1", ".weight_v"), ".parametrizations.weight.original0", ".weight_g"), model_path)
    except Exception as e:
        logger.error(f"{translations['extract_model_error']}: {e}")