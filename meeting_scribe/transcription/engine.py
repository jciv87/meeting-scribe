"""Transcription engine wrapping faster-whisper WhisperModel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WordTimestamp:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class TranscribedSegment:
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = field(default_factory=list)
    audio_chunk: np.ndarray | None = None


class TranscriptionEngine:
    """Wraps faster_whisper.WhisperModel with lazy model loading."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._language = language
        self._beam_size = beam_size
        self._model = None

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def transcribe_chunk(self, audio: np.ndarray) -> list[TranscribedSegment]:
        """Transcribe a single audio chunk and return typed segment objects."""
        segments, _info = self.model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.4,
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 300,
            },
            word_timestamps=True,
            condition_on_previous_text=True,
            no_speech_threshold=0.5,
            initial_prompt="Meeting transcription with multiple speakers.",
            hallucination_silence_threshold=1.0,
        )

        result: list[TranscribedSegment] = []
        for seg in segments:
            words: list[WordTimestamp] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        WordTimestamp(
                            start=w.start,
                            end=w.end,
                            word=w.word,
                            probability=w.probability,
                        )
                    )
            result.append(
                TranscribedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    words=words,
                    audio_chunk=audio,
                )
            )
        return result
