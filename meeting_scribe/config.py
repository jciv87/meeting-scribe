"""Configuration loader for Meeting Scribe."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    device: str = "BlackHole 2ch"
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class TranscriptionConfig:
    model_size: str = "large-v3-turbo"
    compute_type: str = "int8"
    cpu_threads: int = 4
    language: str = "en"
    beam_size: int = 5
    chunk_duration_seconds: int = 30
    overlap_seconds: int = 2


@dataclass
class DiarizationConfig:
    similarity_threshold: float = 0.75
    min_segment_duration: float = 1.0


@dataclass
class DetectionConfig:
    poll_interval_seconds: int = 5
    start_debounce_seconds: int = 3
    end_debounce_seconds: int = 10
    silence_timeout_seconds: int = 60


@dataclass
class HotkeyConfig:
    combination: str = "<fn>+<f12>"


@dataclass
class OutputConfig:
    transcript_dir: str = "~/meeting-scribe/transcripts"
    incremental_save_seconds: int = 60

    @property
    def transcript_path(self) -> Path:
        resolved = Path(os.path.expanduser(self.transcript_dir)).resolve()
        home = Path.home().resolve()
        if not str(resolved).startswith(str(home)):
            raise ValueError(
                f"transcript_dir must be within your home directory, got: {resolved}"
            )
        return resolved


@dataclass
class SummarizationConfig:
    enabled: bool = True
    model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    timeout_seconds: int = 120


@dataclass
class UIConfig:
    auto_detect_meetings: bool = True
    notify_on_detection: bool = True
    auto_start_recording: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def load_config(path: str | None = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, "config.yaml")
        if not os.path.exists(path):
            path = os.path.join(base, "config.yaml.example")

    config = Config()
    if not os.path.exists(path):
        return config

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    section_map = {
        "audio": (AudioConfig, "audio"),
        "transcription": (TranscriptionConfig, "transcription"),
        "diarization": (DiarizationConfig, "diarization"),
        "detection": (DetectionConfig, "detection"),
        "hotkey": (HotkeyConfig, "hotkey"),
        "output": (OutputConfig, "output"),
        "summarization": (SummarizationConfig, "summarization"),
        "ui": (UIConfig, "ui"),
    }

    for key, (cls, attr) in section_map.items():
        if key in data and isinstance(data[key], dict):
            setattr(config, attr, cls(**data[key]))

    return config
