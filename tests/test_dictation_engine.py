"""Tests for meeting_scribe.dictation.engine — pipeline orchestrator."""

import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_scribe.dictation.engine import DictationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(audio_device: str = "") -> tuple[DictationEngine, MagicMock, MagicMock]:
    """Build a DictationEngine with mocked transcription + cleanup."""
    mock_transcription = MagicMock()
    mock_cleanup = MagicMock()
    mock_cleanup.clean.return_value = "CLEANED"  # default return for most tests

    engine = DictationEngine.__new__(DictationEngine)
    engine._engine = mock_transcription
    engine._cleanup = mock_cleanup
    engine._sample_rate = 16000
    engine._audio_device = audio_device
    engine._recording = False
    engine._lock = threading.Lock()
    engine._chunks = []
    engine._stream = None

    return engine, mock_transcription, mock_cleanup


def _fake_segments(texts: list[str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    def test_empty_device_returns_none(self):
        engine, _, _ = _make_engine("")
        assert engine._resolve_device() is None

    @patch("meeting_scribe.dictation.engine.sd.query_devices")
    def test_matching_device_returns_index(self, mock_query):
        mock_query.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "BlackHole 2ch", "max_input_channels": 0},
        ]
        engine, _, _ = _make_engine("built-in")
        assert engine._resolve_device() == 0

    @patch("meeting_scribe.dictation.engine.sd.query_devices")
    def test_no_match_returns_none(self, mock_query):
        mock_query.return_value = [
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]
        engine, _, _ = _make_engine("Yeti")
        assert engine._resolve_device() is None

    @patch("meeting_scribe.dictation.engine.sd.query_devices")
    def test_skips_output_only_devices(self, mock_query):
        mock_query.return_value = [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "Speakers Input", "max_input_channels": 1},
        ]
        engine, _, _ = _make_engine("speakers")
        # Should find the second one (has input channels)
        assert engine._resolve_device() == 1


# ---------------------------------------------------------------------------
# Recording lifecycle
# ---------------------------------------------------------------------------


class TestRecordingLifecycle:
    @patch("meeting_scribe.dictation.engine.sd.InputStream")
    def test_start_sets_recording_flag(self, mock_stream_cls):
        engine, _, _ = _make_engine()
        mock_stream_cls.return_value = MagicMock()
        engine.start_recording()
        assert engine._recording is True

    @patch("meeting_scribe.dictation.engine.sd.InputStream")
    def test_double_start_is_noop(self, mock_stream_cls):
        engine, _, _ = _make_engine()
        mock_stream_cls.return_value = MagicMock()
        engine.start_recording()
        engine.start_recording()
        mock_stream_cls.assert_called_once()

    def test_stop_without_start_is_noop(self):
        engine, _, _ = _make_engine()
        engine.stop_recording()  # should not raise

    @patch("meeting_scribe.dictation.engine.sd.InputStream")
    def test_stop_clears_recording_flag(self, mock_stream_cls):
        engine, _, _ = _make_engine()
        mock_stream_cls.return_value = MagicMock()
        engine.start_recording()
        engine.stop_recording()
        assert engine._recording is False

    @patch("meeting_scribe.dictation.engine.sd.InputStream")
    def test_stop_closes_stream(self, mock_stream_cls):
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream
        engine, _, _ = _make_engine()
        engine.start_recording()
        engine.stop_recording()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------


class TestAudioCallback:
    def test_callback_buffers_mono(self):
        engine, _, _ = _make_engine()
        engine._recording = True
        indata = np.random.randn(1024, 1).astype(np.float32)
        engine._audio_callback(indata, 1024, None, MagicMock())
        assert len(engine._chunks) == 1
        assert engine._chunks[0].shape == (1024,)

    def test_callback_ignores_when_not_recording(self):
        engine, _, _ = _make_engine()
        engine._recording = False
        indata = np.random.randn(1024, 1).astype(np.float32)
        engine._audio_callback(indata, 1024, None, MagicMock())
        assert len(engine._chunks) == 0

    def test_callback_handles_stereo(self):
        engine, _, _ = _make_engine()
        engine._recording = True
        indata = np.random.randn(512, 2).astype(np.float32)
        engine._audio_callback(indata, 512, None, MagicMock())
        assert engine._chunks[0].shape == (512,)

    def test_callback_handles_1d_input(self):
        engine, _, _ = _make_engine()
        engine._recording = True
        indata = np.random.randn(256).astype(np.float32)
        engine._audio_callback(indata, 256, None, MagicMock())
        assert engine._chunks[0].shape == (256,)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TestTranscribe:
    def test_concatenates_segment_texts(self):
        engine, mock_transcription, _ = _make_engine()
        mock_transcription.transcribe_chunk.return_value = _fake_segments(
            ["Hello there.", "How are you?"]
        )
        result = engine._transcribe(np.zeros(16000))
        assert result == "Hello there. How are you?"

    def test_empty_segments_return_empty(self):
        engine, mock_transcription, _ = _make_engine()
        mock_transcription.transcribe_chunk.return_value = []
        result = engine._transcribe(np.zeros(16000))
        assert result == ""

    def test_strips_whitespace_from_segments(self):
        engine, mock_transcription, _ = _make_engine()
        mock_transcription.transcribe_chunk.return_value = _fake_segments(
            ["  hello  ", "  "]
        )
        result = engine._transcribe(np.zeros(16000))
        assert result == "hello"


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------


class TestProcess:
    @patch.object(DictationEngine, "_paste")
    @patch.object(DictationEngine, "_notify")
    def test_full_pipeline(self, mock_notify, mock_paste):
        engine, mock_transcription, mock_cleanup = _make_engine()
        mock_transcription.transcribe_chunk.return_value = _fake_segments(
            ["hello world"]
        )
        mock_cleanup.clean.return_value = "Hello, world."

        engine._process(np.zeros(16000))

        mock_cleanup.clean.assert_called_once_with("hello world")
        mock_paste.assert_called_once_with("Hello, world.")

    @patch.object(DictationEngine, "_paste")
    @patch.object(DictationEngine, "_notify")
    def test_empty_transcription_skips_paste(self, mock_notify, mock_paste):
        engine, mock_transcription, _ = _make_engine()
        mock_transcription.transcribe_chunk.return_value = _fake_segments(["  "])

        engine._process(np.zeros(16000))

        mock_paste.assert_not_called()
        mock_notify.assert_called()

    @patch.object(DictationEngine, "_paste")
    @patch.object(DictationEngine, "_notify")
    def test_exception_does_not_propagate(self, mock_notify, mock_paste):
        engine, mock_transcription, _ = _make_engine()
        mock_transcription.transcribe_chunk.side_effect = RuntimeError("boom")

        engine._process(np.zeros(16000))  # should not raise
        mock_paste.assert_not_called()


# ---------------------------------------------------------------------------
# Paste
# ---------------------------------------------------------------------------


class TestPaste:
    @patch("meeting_scribe.dictation.engine.subprocess.run")
    def test_paste_calls_pbcopy_then_osascript(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        DictationEngine._paste("test text")

        assert mock_run.call_count == 2
        # First call: pbcopy
        first_call = mock_run.call_args_list[0]
        assert first_call.args[0] == ["pbcopy"]
        assert first_call.kwargs["input"] == b"test text"
        # Second call: osascript
        second_call = mock_run.call_args_list[1]
        assert second_call.args[0][0] == "osascript"

    @patch("meeting_scribe.dictation.engine.subprocess.run")
    def test_paste_skips_osascript_if_pbcopy_fails(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr=b"error")
        DictationEngine._paste("test")
        assert mock_run.call_count == 1  # only pbcopy, no osascript


# ---------------------------------------------------------------------------
# Short audio rejection
# ---------------------------------------------------------------------------


class TestShortAudioRejection:
    @patch("meeting_scribe.dictation.engine.sd.InputStream")
    def test_very_short_audio_skipped(self, mock_stream_cls):
        engine, mock_transcription, _ = _make_engine()
        mock_stream_cls.return_value = MagicMock()
        engine.start_recording()

        # Simulate ~0.1s of audio (1600 samples at 16kHz)
        engine._chunks = [np.zeros(1600, dtype=np.float32)]
        engine.stop_recording()
        time.sleep(0.1)

        mock_transcription.transcribe_chunk.assert_not_called()
