"""Tests for meeting_scribe.audio.buffer.AudioRingBuffer."""

import numpy as np
import pytest

from meeting_scribe.audio.buffer import AudioRingBuffer


def make_ramp(n: int, start: float = 0.0) -> np.ndarray:
    """Return n float32 samples counting up from start."""
    return np.arange(start, start + n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Basic write / pending_seconds
# ---------------------------------------------------------------------------

class TestPendingSeconds:
    def test_fresh_buffer_has_zero_pending(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        assert buf.pending_seconds() == 0.0

    def test_pending_increases_after_write(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(50))
        assert buf.pending_seconds() == pytest.approx(0.5)

    def test_pending_capped_at_capacity(self):
        buf = AudioRingBuffer(capacity_seconds=1, sample_rate=100)
        buf.write(make_ramp(300))  # write 3x capacity
        assert buf.pending_seconds() == pytest.approx(1.0)

    def test_pending_accumulates_across_multiple_writes(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(30))
        buf.write(make_ramp(20))
        assert buf.pending_seconds() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# read_pending
# ---------------------------------------------------------------------------

class TestReadPending:
    def test_read_pending_returns_all_samples(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        data = make_ramp(50)
        buf.write(data)
        result = buf.read_pending()
        np.testing.assert_array_equal(result, data)

    def test_read_pending_resets_counter(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(50))
        buf.read_pending()
        assert buf.pending_seconds() == pytest.approx(0.0)

    def test_read_pending_empty_returns_empty_array(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        result = buf.read_pending()
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_read_pending_is_contiguous_after_wraparound(self):
        """Write past capacity so the ring wraps, then read all pending."""
        buf = AudioRingBuffer(capacity_seconds=1, sample_rate=100)
        # Fill to capacity
        buf.write(make_ramp(100))
        buf.read_pending()  # consume everything
        # Write across the seam
        buf.write(make_ramp(60))   # moves write_pos to 60
        buf.write(make_ramp(60, start=60.0))  # wraps around
        result = buf.read_pending()
        # pending is capped at capacity (100), and the content should be float32
        assert len(result) == 100
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# read_chunk with overlap
# ---------------------------------------------------------------------------

class TestReadChunk:
    def test_read_chunk_returns_requested_samples(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(100))
        chunk = buf.read_chunk(duration_seconds=0.5, overlap_seconds=0.0)
        assert len(chunk) == 50

    def test_read_chunk_with_no_overlap_consumes_all(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(100))
        buf.read_chunk(duration_seconds=1.0, overlap_seconds=0.0)
        assert buf.pending_seconds() == pytest.approx(0.0)

    def test_read_chunk_overlap_retains_tail(self):
        """After reading a 1-second chunk with 0.2s overlap, 0.2s remain pending."""
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(100))
        buf.read_chunk(duration_seconds=1.0, overlap_seconds=0.2)
        assert buf.pending_seconds() == pytest.approx(0.2)

    def test_read_chunk_not_enough_data_returns_partial(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(30))
        chunk = buf.read_chunk(duration_seconds=1.0, overlap_seconds=0.0)
        assert len(chunk) == 30

    def test_read_chunk_empty_buffer_returns_empty(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        chunk = buf.read_chunk(duration_seconds=1.0, overlap_seconds=0.0)
        assert len(chunk) == 0

    def test_read_chunk_overlap_larger_than_chunk_keeps_all(self):
        """If overlap >= chunk duration, nothing is consumed."""
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        buf.write(make_ramp(100))
        buf.read_chunk(duration_seconds=1.0, overlap_seconds=1.0)
        # consumed = max(0, 100 - 100) = 0
        assert buf.pending_seconds() == pytest.approx(1.0)

    def test_read_chunk_content_matches_written_data(self):
        buf = AudioRingBuffer(capacity_seconds=5, sample_rate=100)
        data = make_ramp(50)
        buf.write(data)
        chunk = buf.read_chunk(duration_seconds=0.5, overlap_seconds=0.0)
        np.testing.assert_array_equal(chunk, data)


# ---------------------------------------------------------------------------
# Wraparound write integrity
# ---------------------------------------------------------------------------

class TestWraparound:
    def test_write_wraps_without_error(self):
        buf = AudioRingBuffer(capacity_seconds=1, sample_rate=100)
        for _ in range(5):
            buf.write(make_ramp(40))

    def test_write_pos_wraps_correctly(self):
        buf = AudioRingBuffer(capacity_seconds=1, sample_rate=100)
        buf.write(make_ramp(80))
        buf.read_pending()
        buf.write(make_ramp(40))  # crosses the end boundary
        assert buf.pending_seconds() == pytest.approx(0.4)

    def test_float32_dtype_preserved(self):
        buf = AudioRingBuffer(capacity_seconds=2, sample_rate=100)
        buf.write(np.ones(50, dtype=np.int16))
        result = buf.read_pending()
        assert result.dtype == np.float32
