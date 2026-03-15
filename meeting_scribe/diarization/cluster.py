"""Cluster unknown speaker embeddings within a single meeting."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

_LABEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _speaker_label(index: int) -> str:
    """Convert a 0-based cluster index to a human-readable label.

    Supports up to 702 labels (A-Z, then AA-AZ, BA-BZ, ...).
    """
    if index < 26:
        return f"Speaker {_LABEL_LETTERS[index]}"
    outer = index // 26 - 1
    inner = index % 26
    return f"Speaker {_LABEL_LETTERS[outer]}{_LABEL_LETTERS[inner]}"


class MeetingClusterer:
    """Track and cluster unknown speaker embeddings for a single meeting.

    Usage::

        clusterer = MeetingClusterer()
        clusterer.add_unknown(embedding, segment_id=0)
        clusterer.add_unknown(embedding, segment_id=1)
        labels = clusterer.cluster()   # {0: "Speaker A", 1: "Speaker B", ...}
        clusterer.reset()              # prepare for next meeting
    """

    def __init__(self, distance_threshold: float = 0.5) -> None:
        self._distance_threshold = distance_threshold
        self._embeddings: list[np.ndarray] = []
        self._segment_ids: list[int] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_unknown(self, embedding: np.ndarray, segment_id: int) -> None:
        """Store an embedding paired with the segment it came from."""
        self._embeddings.append(embedding.astype(np.float32))
        self._segment_ids.append(segment_id)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster(self) -> dict[int, str]:
        """Cluster stored embeddings and return a segment-id -> label map.

        Returns an empty dict when fewer than two embeddings have been added
        (clustering is undefined for a single point; that segment gets
        "Speaker A" by convention).
        """
        n = len(self._embeddings)
        if n == 0:
            return {}

        if n == 1:
            return {self._segment_ids[0]: _speaker_label(0)}

        matrix = np.vstack(self._embeddings)

        # Pairwise cosine distances then agglomerative linkage
        distances = pdist(matrix, metric="cosine")
        Z = linkage(distances, method="average")

        raw_labels: np.ndarray = fcluster(
            Z, t=self._distance_threshold, criterion="distance"
        )

        # fcluster returns 1-based integers; re-map to consecutive 0-based
        unique_raw = sorted(set(raw_labels.tolist()))
        raw_to_idx = {raw: idx for idx, raw in enumerate(unique_raw)}

        return {
            seg_id: _speaker_label(raw_to_idx[int(raw_label)])
            for seg_id, raw_label in zip(self._segment_ids, raw_labels)
        }

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored embeddings to prepare for a new meeting."""
        self._embeddings.clear()
        self._segment_ids.clear()
