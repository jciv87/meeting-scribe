"""Match live audio segments to known speaker profiles."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine

from .embeddings import SpeakerEncoder
from .profiles import ProfileStore, SpeakerProfile

_UNKNOWN = "Unknown"


class SpeakerMatcher:
    """Identify speakers in audio segments by comparing against stored profiles.

    Similarity is measured with cosine similarity (1 - cosine distance).
    A match is accepted when similarity >= threshold.
    """

    def __init__(
        self,
        profile_store: ProfileStore,
        encoder: SpeakerEncoder,
        threshold: float = 0.75,
    ) -> None:
        self._store = profile_store
        self._encoder = encoder
        self._threshold = threshold

        # Cache: speaker_id -> (name, centroid)
        self._cache: dict[int, tuple[str, np.ndarray]] = {}
        self.refresh_profiles()

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def refresh_profiles(self) -> None:
        """Reload all profiles from the store and rebuild the centroid cache."""
        self._cache.clear()
        profiles: list[SpeakerProfile] = self._store.get_all_profiles()
        for profile in profiles:
            if profile.embeddings:
                self._cache[profile.id] = (profile.name, profile.centroid())

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def identify(self, audio_segment: np.ndarray) -> tuple[str, float]:
        """Identify the speaker in an audio segment.

        Args:
            audio_segment: 1-D float32 numpy array of audio samples at 16 kHz.

        Returns:
            (speaker_name, confidence) where confidence is cosine similarity
            in [0, 1].  Returns ("Unknown", 0.0) when no profile matches.
        """
        if not self._cache:
            return (_UNKNOWN, 0.0)

        embedding = self._encoder.extract(audio_segment)

        best_name = _UNKNOWN
        best_similarity = 0.0

        for name, centroid in self._cache.values():
            distance = cosine(embedding, centroid)
            # cosine() returns distance in [0, 2]; similarity = 1 - distance
            similarity = float(1.0 - distance)
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        if best_similarity >= self._threshold:
            return (best_name, best_similarity)

        return (_UNKNOWN, 0.0)
