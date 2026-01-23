from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union
from txtsplit import txtsplit
from tqdm import tqdm


class TTSModel(ABC):
    """Base class for all TTS models."""

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: The text to synthesize
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple containing:
                - Audio array as numpy array
                - Sample rate as integer
        """
        pass

    def longform(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Synthesize long text by splitting into chunks and concatenating results.

        Args:
            text: The text to synthesize
            **kwargs: Additional arguments passed to synthesize()

        Returns:
            Tuple containing:
                - Concatenated audio array as numpy array
                - Sample rate as integer
        """
        chunks = txtsplit(text)

        # Synthesize each chunk
        audio_chunks = []
        sr = None
        for chunk in tqdm(chunks):
            audio, chunk_sr = self.synthesize(chunk, **kwargs)
            if sr is None:
                sr = chunk_sr
            elif sr != chunk_sr:
                raise ValueError("Inconsistent sample rates between chunks")
            audio_chunks.append(audio)

        # Concatenate audio chunks
        return np.concatenate(audio_chunks), sr

    @abstractmethod
    def __init__(self, device: Optional[str] = "auto", **kwargs):
        """
        Initialize the model.

        Args:
            device: Device to run model on ('cpu', 'cuda', or 'auto')
            **kwargs: Additional model-specific arguments
        """
        pass


def requires_package(package: str):
    raise ImportError(
        f"{package} is not installed. Please install it using `pip install {package}`."
    )


def requires_extra(package: str):
    raise ImportError(
        f"You need to install the {package} extra to use this model. Please install it using `pip install simpletts[{package}]`."
    )
