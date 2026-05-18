import json
import pickle

default_param = {}
default_param["bins"] = -1
default_param["unstable_bins"] = -1
default_param["stable_bins"] = -1
default_param["sr"] = 44100
default_param["pre_filter_start"] = -1
default_param["pre_filter_stop"] = -1
default_param["band"] = {}

N_BINS = "n_bins"

def int_keys(pairs):
    result_dict = {}

    for key, value in pairs:
        if isinstance(key, str) and key.isdigit(): key = int(key)
        result_dict[key] = value

    return result_dict

class ModelParameters(object):
    def __init__(self, config_path="", key_in_bin=None):
        if config_path.endswith(".bin"):
            with open(config_path, "rb") as f:
                data = pickle.load(f)
                self.param = data[key_in_bin]
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                self.param = json.loads(f.read(), object_pairs_hook=int_keys)

        for k in ["mid_side", "mid_side_b", "mid_side_b2", "stereo_w", "stereo_n", "reverse"]:
            if k not in self.param:
                self.param[k] = False

        if N_BINS in self.param:
            self.param["bins"] = self.param[N_BINS]