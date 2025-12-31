from torch import nn
from transformers import HubertModel

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    def extract_features(self, source, padding_mask = None, output_layer = None):
        return self.forward(source)