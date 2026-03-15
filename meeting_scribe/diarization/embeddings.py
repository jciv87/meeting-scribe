"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN model.

torch and speechbrain are imported lazily inside SpeakerEncoder so that the
module itself can be imported without those heavy packages installed.  The
ImportError surfaces only when SpeakerEncoder is actually instantiated or
when extract() is first called.
"""

from __future__ import annotations

import threading

import numpy as np

_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_EMBEDDING_DIM = 192


def _import_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for speaker embeddings. "
            "Install it with: pip install torch"
        ) from e


def _import_encoder_classifier():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        return EncoderClassifier
    except ImportError:
        pass
    try:
        from speechbrain.pretrained import EncoderClassifier
        return EncoderClassifier
    except ImportError as e:
        raise ImportError(
            "speechbrain is required for speaker embeddings. "
            "Install it with: pip install speechbrain>=1.0"
        ) from e


class SpeakerEncoder:
    """Lazy-loading speaker encoder backed by SpeechBrain ECAPA-TDNN.

    Thread-safe: a lock guards all model inference calls because SpeechBrain
    is not guaranteed to be thread-safe.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            EncoderClassifier = _import_encoder_classifier()
            self._model = EncoderClassifier.from_hparams(
                source=_MODEL_SOURCE,
                run_opts={"device": "cpu"},
            )
        return self._model

    def extract(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dim speaker embedding from an audio segment.

        Args:
            audio_segment: 1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (must be 16000 for ECAPA).

        Returns:
            numpy array of shape (192,).
        """
        torch = _import_torch()

        if audio_segment.ndim != 1:
            raise ValueError(
                f"audio_segment must be 1-D, got shape {audio_segment.shape}"
            )

        waveform = torch.from_numpy(audio_segment.astype(np.float32)).unsqueeze(0)
        lengths = torch.tensor([1.0])

        with self._lock:
            model = self._load_model()
            with torch.no_grad():
                embedding = model.encode_batch(waveform, lengths)

        # embedding shape: (batch, 1, dim) — squeeze to (dim,)
        result: np.ndarray = embedding.squeeze().cpu().numpy()
        return result
