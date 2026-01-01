import torch
import onnxruntime

class HubertModelONNX:
    def __init__(self, embedder_model_path, providers, device):
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        self.model = onnxruntime.InferenceSession(embedder_model_path, sess_options=sess_options, providers=providers)
        self.final_proj = self._final_proj
        self.device = device

    def _final_proj(self, source):
        return source
    
    def extract_features(self, source, padding_mask = None, output_layer = None):
        logits = self.model.run([self.model.get_outputs()[0].name, self.model.get_outputs()[1].name], {"feats": source.detach().cpu().numpy()})
        return [torch.as_tensor(logits[int(output_layer != 9)], dtype=torch.float32, device=self.device)]