"""Threaded transcription worker with overlap deduplication."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .engine import TranscriptionEngine, TranscribedSegment


class TranscriptionWorker:
    """Runs transcription in a dedicated daemon thread.

    Pulls audio chunks from *chunk_queue*, transcribes each one via the
    provided engine, adjusts timestamps by a running *chunk_offset*, and
    pushes the resulting ``list[TranscribedSegment]`` onto *result_queue*.

    Overlap deduplication is applied at word level: if a word from the
    incoming chunk starts within the overlap window AND its text matches one
    of the last few words of the previous chunk, it is dropped before the
    adjusted segments are enqueued.
    """

    # How many trailing words of the previous chunk to compare against.
    _OVERLAP_TAIL = 8
    # A word is considered duplicated if it starts within this many seconds
    # of the end of the previous chunk's last word.
    _OVERLAP_WINDOW_S: float = 1.0

    def __init__(
        self,
        engine: TranscriptionEngine,
        chunk_queue: queue.Queue,
        result_queue: queue.Queue,
        sample_rate: int = 16000,
    ) -> None:
        self._engine = engine
        self._chunk_queue = chunk_queue
        self._result_queue = result_queue
        self._sample_rate = sample_rate

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Cumulative time offset (seconds) for the start of the next chunk.
        self.chunk_offset: float = 0.0

        # Trailing words from the most recently processed chunk (absolute ts).
        self._prev_tail_words: list[tuple[float, str]] = []  # (start_abs, word)
        self._prev_tail_end: float = 0.0  # absolute end time of last word

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create and start the daemon thread."""
        self._thread = threading.Thread(target=self.run, daemon=True, name="TranscriptionWorker")
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop: pull chunks, transcribe, deduplicate, enqueue results."""
        while not self._stop_event.is_set():
            try:
                chunk: tuple[object, float] = self._chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio, chunk_duration = chunk  # caller provides (ndarray, duration_s)

            try:
                raw_segments = self._engine.transcribe_chunk(audio)
            except Exception:
                # Do not crash the worker on a bad chunk; just skip it.
                self._chunk_queue.task_done()
                self.chunk_offset += chunk_duration
                continue

            adjusted = self._adjust_and_deduplicate(raw_segments)

            if adjusted:
                self._result_queue.put(adjusted)

            # Advance the offset by the actual chunk duration.
            self.chunk_offset += chunk_duration
            self._chunk_queue.task_done()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjust_and_deduplicate(
        self, segments: list[TranscribedSegment]
    ) -> list[TranscribedSegment]:
        """Shift segment/word timestamps by chunk_offset and strip duplicates."""
        offset = self.chunk_offset
        tail_words = self._prev_tail_words
        overlap_window = self._OVERLAP_WINDOW_S

        adjusted_segments: list[TranscribedSegment] = []

        for seg in segments:
            abs_start = seg.start + offset
            abs_end = seg.end + offset

            filtered_words = []
            for w in seg.words:
                abs_w_start = w.start + offset

                # Deduplication: drop if within overlap window AND text matches
                # a word in the recent tail of the previous chunk.
                if abs_w_start <= self._prev_tail_end + overlap_window:
                    norm = w.word.strip().lower()
                    if any(norm == t.strip().lower() for _, t in tail_words):
                        continue

                from .engine import WordTimestamp
                filtered_words.append(
                    WordTimestamp(
                        start=abs_w_start,
                        end=w.end + offset,
                        word=w.word,
                        probability=w.probability,
                    )
                )

            adjusted_segments.append(
                TranscribedSegment(
                    start=abs_start,
                    end=abs_end,
                    text=seg.text,
                    words=filtered_words,
                    audio_chunk=seg.audio_chunk,
                )
            )

        # Update tail state from all words in this batch.
        all_words = [w for s in adjusted_segments for w in s.words]
        if all_words:
            self._prev_tail_words = [
                (w.start, w.word) for w in all_words[-self._OVERLAP_TAIL :]
            ]
            self._prev_tail_end = all_words[-1].end
        elif adjusted_segments:
            self._prev_tail_end = adjusted_segments[-1].end

        return adjusted_segments
