import os
import sys
import json
import torch
import hashlib
import datetime

from collections import OrderedDict


from arvc.utils.variables import logger, translations, config
from arvc.engine.models.weight_norm import convert_new_to_old

def extract_model(
    ckpt,
    sr,
    pitch_guidance,
    name,
    model_path,
    epoch,
    step,
    version,
    hps,
    model_author,
    vocoder,
    energy_use,
    speakers_id,
    architecture,
    embedder_model=None,
    dataset_length=None,
    overtrain_info="",
):
    """Extract a deployable .pth from a training checkpoint.

    ACCURACY PATCH (Applio parity): added 3 optional kwargs:
      - `embedder_model`: which HuBERT/contentvec variant was used during
        training. Lets inference auto-select the matching embedder instead
        of falling back to defaults — important for accuracy when a non-
        default embedder was used at training time.
      - `dataset_length`: total training audio duration (provenance).
      - `overtrain_info`: overtraining detector summary string.
    All three are persisted into the .pth metadata and are backward-
    compatible (default to None / "" — older loaders just ignore them).
    """
    try:
        logger.info(translations["savemodel"].format(model_dir=model_path, epoch=epoch, step=step))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        opt = OrderedDict(weight={key: (value if not config.device.startswith("privateuseone") else value.detach().cpu()).to(torch.float16 if config.is_half else torch.float32) for key, value in ckpt.items() if "enc_q" not in key})
        opt["config"] = [hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, hps.model.inter_channels, hps.model.hidden_channels, hps.model.filter_channels, hps.model.n_heads, hps.model.n_layers, hps.model.kernel_size, hps.model.p_dropout, hps.model.resblock, hps.model.resblock_kernel_sizes, hps.model.resblock_dilation_sizes, hps.model.upsample_rates, hps.model.upsample_initial_channel, hps.model.upsample_kernel_sizes, hps.model.spk_embed_dim, hps.model.gin_channels, hps.data.sample_rate]
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
        opt["speakers_id"] = speakers_id
        opt["architecture"] = architecture

        # ACCURACY PATCH (Applio parity): provenance fields.
        # `embedder_model` is the most important of the three — lets inference
        # pick the matching embedder automatically. `dataset_length` and
        # `overtrain_info` are user-facing metadata only.
        if embedder_model is not None:
            opt["embedder_model"] = embedder_model
        if dataset_length is not None:
            opt["dataset_length"] = dataset_length
        if overtrain_info:
            opt["overtrain_info"] = overtrain_info

        torch.save(
            convert_new_to_old(opt),
            model_path
        )
    except Exception as e:
        logger.error(f"{translations['extract_model_error']}: {e}")