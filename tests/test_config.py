"""Tests for meeting_scribe.config — load_config and all dataclass defaults."""

import pytest
import yaml
from pathlib import Path

from meeting_scribe.config import (
    Config,
    AudioConfig,
    TranscriptionConfig,
    DiarizationConfig,
    DetectionConfig,
    HotkeyConfig,
    OutputConfig,
    SummarizationConfig,
    UIConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_audio_defaults(self):
        a = AudioConfig()
        assert a.device == "BlackHole 2ch"
        assert a.sample_rate == 16000
        assert a.channels == 1

    def test_transcription_defaults(self):
        t = TranscriptionConfig()
        assert t.model_size == "large-v3-turbo"
        assert t.compute_type == "int8"
        assert t.cpu_threads == 4
        assert t.language == "en"
        assert t.beam_size == 5
        assert t.chunk_duration_seconds == 30
        assert t.overlap_seconds == 2

    def test_diarization_defaults(self):
        d = DiarizationConfig()
        assert d.similarity_threshold == 0.75
        assert d.min_segment_duration == 1.0

    def test_detection_defaults(self):
        d = DetectionConfig()
        assert d.poll_interval_seconds == 5
        assert d.start_debounce_seconds == 3
        assert d.end_debounce_seconds == 10
        assert d.silence_timeout_seconds == 60

    def test_hotkey_defaults(self):
        h = HotkeyConfig()
        assert h.combination == "<fn>+<f12>"

    def test_output_defaults(self):
        o = OutputConfig()
        assert o.transcript_dir == "~/meeting-scribe/transcripts"
        assert o.incremental_save_seconds == 60

    def test_summarization_defaults(self):
        s = SummarizationConfig()
        assert s.enabled is True
        assert s.model == "llama3.1:8b"
        assert s.ollama_host == "http://localhost:11434"
        assert s.timeout_seconds == 120

    def test_ui_defaults(self):
        u = UIConfig()
        assert u.auto_detect_meetings is True
        assert u.notify_on_detection is True
        assert u.auto_start_recording is False

    def test_config_has_all_sections(self, default_config):
        cfg = default_config
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.transcription, TranscriptionConfig)
        assert isinstance(cfg.diarization, DiarizationConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.hotkey, HotkeyConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.summarization, SummarizationConfig)
        assert isinstance(cfg.ui, UIConfig)


# ---------------------------------------------------------------------------
# load_config — missing file
# ---------------------------------------------------------------------------

class TestLoadConfigMissingFile:
    def test_missing_path_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, Config)
        assert cfg.audio.device == "BlackHole 2ch"

    def test_missing_path_summarization_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "no.yaml"))
        assert cfg.summarization.enabled is True
        assert cfg.transcription.model_size == "large-v3-turbo"


# ---------------------------------------------------------------------------
# load_config — full YAML
# ---------------------------------------------------------------------------

class TestLoadConfigFullYAML:
    def test_full_yaml_overrides_all_sections(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        data = {
            "audio": {"device": "Loopback", "sample_rate": 44100, "channels": 2},
            "transcription": {"model_size": "medium", "language": "fr"},
            "summarization": {"enabled": False, "model": "mistral"},
        }
        config_file.write_text(yaml.dump(data))

        cfg = load_config(str(config_file))

        assert cfg.audio.device == "Loopback"
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.channels == 2
        assert cfg.transcription.model_size == "medium"
        assert cfg.transcription.language == "fr"
        assert cfg.summarization.enabled is False
        assert cfg.summarization.model == "mistral"

    def test_full_yaml_unmentioned_sections_keep_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"audio": {"device": "Test"}}))

        cfg = load_config(str(config_file))

        assert cfg.transcription.model_size == "large-v3-turbo"
        assert cfg.diarization.similarity_threshold == 0.75


# ---------------------------------------------------------------------------
# load_config — partial YAML
# ---------------------------------------------------------------------------

class TestLoadConfigPartialYAML:
    def test_partial_yaml_only_overrides_present_keys(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"output": {"incremental_save_seconds": 30}}))

        cfg = load_config(str(config_file))

        assert cfg.output.incremental_save_seconds == 30
        assert cfg.output.transcript_dir == "~/meeting-scribe/transcripts"

    def test_empty_yaml_file_returns_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        cfg = load_config(str(config_file))
        assert cfg.audio.device == "BlackHole 2ch"

    def test_non_dict_section_is_ignored(self, tmp_path):
        """A YAML section with a non-dict value must not crash; defaults survive."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"audio": "not-a-dict"}))

        cfg = load_config(str(config_file))
        assert cfg.audio.device == "BlackHole 2ch"


# ---------------------------------------------------------------------------
# OutputConfig.transcript_path property
# ---------------------------------------------------------------------------

class TestOutputConfigTranscriptPath:
    def test_default_path_is_absolute(self):
        o = OutputConfig()
        assert o.transcript_path.is_absolute()

    def test_path_outside_home_raises(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(
            "meeting_scribe.config.Path.home",
            classmethod(lambda cls: fake_home),
        )
        o = OutputConfig(transcript_dir="/tmp/evil")
        with pytest.raises(ValueError, match="home directory"):
            _ = o.transcript_path
