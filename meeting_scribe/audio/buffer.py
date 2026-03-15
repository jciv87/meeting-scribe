import threading

import numpy as np


class AudioRingBuffer:
    """Pre-allocated circular buffer for float32 audio samples."""

    def __init__(self, capacity_seconds: int = 300, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._capacity = capacity_seconds * sample_rate
        self._buf: np.ndarray = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._pending_samples: int = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Write samples into the ring buffer, overwriting oldest data if full."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        n = len(samples)
        with self._lock:
            space_to_end = self._capacity - self._write_pos
            if n <= space_to_end:
                self._buf[self._write_pos : self._write_pos + n] = samples
            else:
                self._buf[self._write_pos :] = samples[:space_to_end]
                remainder = n - space_to_end
                self._buf[:remainder] = samples[space_to_end:]
            self._write_pos = (self._write_pos + n) % self._capacity
            self._pending_samples = min(self._pending_samples + n, self._capacity)

    def pending_seconds(self) -> float:
        """Return the number of seconds of unconsumed (pending) audio."""
        with self._lock:
            return self._pending_samples / self._sample_rate

    def read_pending(self) -> np.ndarray:
        """Return all pending audio as a contiguous array and reset the pending counter."""
        with self._lock:
            n = self._pending_samples
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                data = self._buf[start : start + n].copy()
            else:
                first = self._capacity - start
                data = np.concatenate(
                    [self._buf[start:], self._buf[: n - first]]
                )
            self._pending_samples = 0
            return data

    def read_chunk(self, duration_seconds: float, overlap_seconds: float) -> np.ndarray:
        """Return a chunk of *duration_seconds* and keep *overlap_seconds* as pending.

        The returned array has exactly ``duration_seconds * sample_rate`` samples
        (or fewer if not enough data is available). The overlap is retained so
        the next chunk begins with that tail of audio.
        """
        with self._lock:
            chunk_samples = int(duration_seconds * self._sample_rate)
            overlap_samples = int(overlap_seconds * self._sample_rate)
            n = min(self._pending_samples, chunk_samples)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write_pos - self._pending_samples) % self._capacity
            if start + n <= self._capacity:
                data = self._buf[start : start + n].copy()
            else:
                first = self._capacity - start
                data = np.concatenate(
                    [self._buf[start:], self._buf[: n - first]]
                )
            consumed = max(0, n - overlap_samples)
            self._pending_samples -= consumed
            return data
