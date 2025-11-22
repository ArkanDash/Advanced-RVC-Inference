"""RVC Applio compatibility package."""

try:
    from .infer.infer import VoiceConverter
    from .infer.pipeline import RVC_Inference_Pipeline
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False
    VoiceConverter = None
    RVC_Inference_Pipeline = None

__all__ = [
    "VoiceConverter",
    "RVC_Inference_Pipeline", 
    "RVC_AVAILABLE"
]