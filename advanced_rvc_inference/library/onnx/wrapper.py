import json
import onnx
import torch
import onnxruntime

import numpy as np

class ONNXRVC:
    def __init__(self, model_path, providers, log_severity_level = 3):
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = log_severity_level

        metadata_dict = None
        for prop in onnx.load(model_path).metadata_props:
            if prop.key == "model_info":
                metadata_dict = json.loads(prop.value)
                break

        self.cpt = {}
        self.cpt["tgt_sr"] = metadata_dict.get("sr", 32000)
        self.cpt["use_f0"] = metadata_dict.get("f0", 1)
        self.cpt["version"] = metadata_dict.get("version", "v1")
        self.cpt["energy"] = metadata_dict.get("energy", False)
        self.net_g = onnxruntime.InferenceSession(
            model_path, 
            sess_options=sess_options, 
            providers=providers
        )

    def get_onnx_argument(self, feats, p_len, sid, pitch, pitchf, energy):
        inputs = {
            self.net_g.get_inputs()[0].name: feats.cpu().numpy().astype(np.float32),
            self.net_g.get_inputs()[1].name: p_len.cpu().numpy(),
            self.net_g.get_inputs()[2].name: np.array([sid.cpu().item()], dtype=np.int64),
            self.net_g.get_inputs()[3].name: np.random.randn(1, 192, p_len).astype(np.float32)
        }

        if self.cpt["energy"]:
            if self.cpt["use_f0"]:
                inputs.update({
                    self.net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64),
                    self.net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32),
                    self.net_g.get_inputs()[6].name: energy.cpu().numpy().astype(np.float32)
                })
            else:
                inputs.update({
                    self.net_g.get_inputs()[4].name: energy.cpu().numpy().astype(np.float32)
                })
        else:
            if self.cpt["use_f0"]:
                inputs.update({
                    self.net_g.get_inputs()[4].name: pitch.cpu().numpy().astype(np.int64),
                    self.net_g.get_inputs()[5].name: pitchf.cpu().numpy().astype(np.float32)
                })

        return inputs
    
    def to(self, device = "cpu"):
        self.device = device
        return self

    def infer(self, feats = None, p_len = None, pitch = None, pitchf = None, sid = None, energy = None):
        output = self.net_g.run(
            [self.net_g.get_outputs()[0].name], (
                self.get_onnx_argument(
                    feats, 
                    p_len, 
                    sid, 
                    pitch, 
                    pitchf, 
                    energy, 
                )
            )
        )

        return torch.as_tensor(output, device=self.device)
