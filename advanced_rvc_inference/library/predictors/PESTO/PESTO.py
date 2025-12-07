import os
import sys
import torch

sys.path.append(os.getcwd())

class PESTO:
    def __init__(self, model_path, step_size=10, reduction="alwa", num_chunks=1, sample_rate=16000, device=None, providers=None, onnx=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.step_size = step_size
        self.reduction = reduction
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PESTO.model import PPESTO, Resnet1d
            from main.library.predictors.PESTO.preprocessor import Preprocessor

            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            model = PPESTO(Resnet1d(**ckpt["hparams"]["encoder"]), preprocessor=Preprocessor(hop_size=step_size, sampling_rate=sample_rate, **ckpt["hcqt_params"]), crop_kwargs=ckpt["hparams"]["pitch_shift"], reduction=ckpt["hparams"]["reduction"])
            model.load_state_dict(ckpt["state_dict"], strict=False)

            self.model = model.to(self.device).eval()
            self.model.reduction = self.reduction

    def compute_f0(self, x):
        assert x.ndim <= 2

        with torch.inference_mode():
            with torch.no_grad():
                preds, confidence = [], []

                for chunk in x.chunk(chunks=self.num_chunks):
                    if self.onnx:
                        model = self.model.run(
                            [self.model.get_outputs()[0].name, self.model.get_outputs()[1].name], 
                            {
                                self.model.get_inputs()[0].name: chunk.cpu().numpy()
                            }
                        )
                        pred, conf = torch.tensor(model[0], device=self.device), torch.tensor(model[1], device=self.device)
                    else:
                        pred, conf = self.model(
                            chunk, 
                            sr=self.sample_rate, 
                            convert_to_freq=True, 
                            return_activations=False
                        )

                    preds.append(pred)
                    confidence.append(conf)

                return torch.cat(preds, dim=0), torch.cat(confidence, dim=0)