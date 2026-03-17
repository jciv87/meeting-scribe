"""Shared fixtures for meeting-scribe test suite."""

import pytest
from datetime import datetime
from pathlib import Path

from meeting_scribe.config import Config, AudioConfig, OutputConfig, SummarizationConfig
from meeting_scribe.output.markdown import MarkdownTranscript


@pytest.fixture
def tmp_transcript_dir(tmp_path: Path) -> Path:
    """A temporary directory for transcript output."""
    d = tmp_path / "transcripts"
    d.mkdir()
    return d


@pytest.fixture
def default_config() -> Config:
    """A Config instance with all defaults."""
    return Config()


@pytest.fixture
def sample_start_time() -> datetime:
    return datetime(2024, 6, 15, 10, 0, 0)


@pytest.fixture
def sample_transcript(sample_start_time: datetime) -> MarkdownTranscript:
    """A MarkdownTranscript pre-populated with two entries."""
    t = MarkdownTranscript(source="Zoom", start_time=sample_start_time)
    t.add_entry(0.0, "Alice", "Hello everyone.")
    t.add_entry(5.0, "Bob", "Hi there.")
    return t
