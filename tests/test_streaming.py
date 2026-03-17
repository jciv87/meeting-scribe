"""Tests for TranscriptionWorker deduplication logic (_adjust_and_deduplicate)."""

import pytest
import queue

from meeting_scribe.transcription.engine import TranscribedSegment, WordTimestamp
from meeting_scribe.transcription.streaming import TranscriptionWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_word(start: float, end: float, text: str, prob: float = 0.9) -> WordTimestamp:
    return WordTimestamp(start=start, end=end, word=text, probability=prob)


def make_segment(
    start: float,
    end: float,
    text: str,
    words: list[WordTimestamp] | None = None,
) -> TranscribedSegment:
    return TranscribedSegment(
        start=start,
        end=end,
        text=text,
        words=words or [],
        audio_chunk=None,
    )


def make_worker() -> TranscriptionWorker:
    """Worker with no real engine — only _adjust_and_deduplicate is tested."""
    return TranscriptionWorker(
        engine=None,  # type: ignore[arg-type]
        chunk_queue=queue.Queue(),
        result_queue=queue.Queue(),
        sample_rate=16000,
    )


# ---------------------------------------------------------------------------
# Timestamp adjustment
# ---------------------------------------------------------------------------

class TestTimestampAdjustment:
    def test_segment_timestamps_shifted_by_offset(self):
        worker = make_worker()
        worker.chunk_offset = 10.0
        seg = make_segment(0.0, 2.0, "Hello")
        result = worker._adjust_and_deduplicate([seg])
        assert result[0].start == pytest.approx(10.0)
        assert result[0].end == pytest.approx(12.0)

    def test_word_timestamps_shifted_by_offset(self):
        worker = make_worker()
        worker.chunk_offset = 5.0
        words = [make_word(0.0, 0.5, "Hello"), make_word(0.5, 1.0, "world")]
        seg = make_segment(0.0, 1.0, "Hello world", words=words)
        result = worker._adjust_and_deduplicate([seg])
        assert result[0].words[0].start == pytest.approx(5.0)
        assert result[0].words[1].start == pytest.approx(5.5)

    def test_zero_offset_leaves_timestamps_unchanged(self):
        worker = make_worker()
        words = [make_word(1.0, 1.5, "Test")]
        seg = make_segment(1.0, 1.5, "Test", words=words)
        result = worker._adjust_and_deduplicate([seg])
        assert result[0].start == pytest.approx(1.0)
        assert result[0].words[0].start == pytest.approx(1.0)

    def test_empty_segments_returns_empty(self):
        worker = make_worker()
        assert worker._adjust_and_deduplicate([]) == []


# ---------------------------------------------------------------------------
# Deduplication: duplicate words dropped
# ---------------------------------------------------------------------------

class TestDeduplication:
    def _prime_tail(self, worker: TranscriptionWorker, words: list[WordTimestamp]) -> None:
        """Run a first chunk through the worker to set _prev_tail_words."""
        seg = make_segment(words[0].start, words[-1].end, "priming", words=words)
        worker._adjust_and_deduplicate([seg])

    def test_duplicate_word_in_overlap_window_is_dropped(self):
        worker = make_worker()
        # First chunk ends at t=5.0 with word "hello" at t=4.5
        first_words = [make_word(4.0, 4.5, "hello")]
        self._prime_tail(worker, first_words)

        # Second chunk — "hello" starts within overlap window of prev tail end
        worker.chunk_offset = 4.0
        dup_word = make_word(0.0, 0.5, "hello")   # abs start = 4.0, tail_end = 4.5, window = 1.0 → within
        new_word = make_word(1.0, 1.5, "world")
        seg = make_segment(0.0, 1.5, "hello world", words=[dup_word, new_word])
        result = worker._adjust_and_deduplicate([seg])
        word_texts = [w.word for w in result[0].words]
        assert "hello" not in word_texts
        assert "world" in word_texts

    def test_non_duplicate_word_not_dropped(self):
        worker = make_worker()
        first_words = [make_word(4.0, 4.5, "hello")]
        self._prime_tail(worker, first_words)

        worker.chunk_offset = 5.5
        new_word = make_word(0.0, 0.5, "world")   # abs start = 5.5, outside overlap window
        seg = make_segment(0.0, 0.5, "world", words=[new_word])
        result = worker._adjust_and_deduplicate([seg])
        assert result[0].words[0].word == "world"

    def test_case_insensitive_dedup(self):
        worker = make_worker()
        first_words = [make_word(4.0, 4.5, "Hello")]
        self._prime_tail(worker, first_words)

        worker.chunk_offset = 4.0
        dup_word = make_word(0.0, 0.5, "hello")  # lowercase version
        seg = make_segment(0.0, 0.5, "hello", words=[dup_word])
        result = worker._adjust_and_deduplicate([seg])
        assert result[0].words == []

    def test_word_outside_overlap_window_not_deduped(self):
        worker = make_worker()
        first_words = [make_word(0.0, 0.5, "hello")]
        self._prime_tail(worker, first_words)

        # abs start = 0 + 10.0 = 10.0; tail_end = 0.5; window = 1.0 → 10.0 > 1.5 → NOT dup
        worker.chunk_offset = 10.0
        word = make_word(0.0, 0.5, "hello")
        seg = make_segment(0.0, 0.5, "hello", words=[word])
        result = worker._adjust_and_deduplicate([seg])
        assert len(result[0].words) == 1


# ---------------------------------------------------------------------------
# Tail state update
# ---------------------------------------------------------------------------

class TestTailStateUpdate:
    def test_tail_words_updated_after_chunk(self):
        worker = make_worker()
        words = [make_word(0.0, 0.5, "one"), make_word(0.5, 1.0, "two")]
        seg = make_segment(0.0, 1.0, "one two", words=words)
        worker._adjust_and_deduplicate([seg])
        tail_texts = [w for _, w in worker._prev_tail_words]
        assert "one" in tail_texts
        assert "two" in tail_texts

    def test_tail_capped_at_overlap_tail_constant(self):
        worker = make_worker()
        many_words = [make_word(float(i), float(i) + 0.5, f"w{i}") for i in range(20)]
        seg = make_segment(0.0, 10.0, "many words", words=many_words)
        worker._adjust_and_deduplicate([seg])
        assert len(worker._prev_tail_words) <= worker._OVERLAP_TAIL

    def test_tail_end_updated_to_last_word_end(self):
        worker = make_worker()
        words = [make_word(0.0, 0.5, "a"), make_word(0.5, 1.2, "b")]
        seg = make_segment(0.0, 1.2, "a b", words=words)
        worker._adjust_and_deduplicate([seg])
        assert worker._prev_tail_end == pytest.approx(1.2)

    def test_no_words_in_segments_does_not_crash(self):
        worker = make_worker()
        seg = make_segment(0.0, 2.0, "silence")  # no words
        result = worker._adjust_and_deduplicate([seg])
        assert len(result) == 1
