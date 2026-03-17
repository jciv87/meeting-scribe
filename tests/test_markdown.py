"""Tests for meeting_scribe.output.markdown.MarkdownTranscript."""

import pytest
from datetime import datetime, timedelta

from meeting_scribe.output.markdown import MarkdownTranscript, TranscriptEntry


def make_transcript(source="Zoom") -> MarkdownTranscript:
    start = datetime(2024, 3, 15, 9, 0, 0)
    return MarkdownTranscript(source=source, start_time=start)


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------

class TestAddEntry:
    def test_add_single_entry(self):
        t = make_transcript()
        t.add_entry(0.0, "Alice", "Hello.")
        assert len(t._entries) == 1

    def test_add_multiple_entries_preserves_order(self):
        t = make_transcript()
        t.add_entry(0.0, "Alice", "First.")
        t.add_entry(5.0, "Bob", "Second.")
        t.add_entry(10.0, "Alice", "Third.")
        assert [e.speaker for e in t._entries] == ["Alice", "Bob", "Alice"]

    def test_entry_fields_stored_correctly(self):
        t = make_transcript()
        t.add_entry(3.5, "Carol", "Some text.")
        e = t._entries[0]
        assert e.timestamp == 3.5
        assert e.speaker == "Carol"
        assert e.text == "Some text."


# ---------------------------------------------------------------------------
# set_end_time
# ---------------------------------------------------------------------------

class TestSetEndTime:
    def test_set_end_time_stores_value(self):
        t = make_transcript()
        end = datetime(2024, 3, 15, 9, 45, 0)
        t.set_end_time(end)
        assert t.end_time == end

    def test_end_time_defaults_to_none(self):
        t = make_transcript()
        assert t.end_time is None


# ---------------------------------------------------------------------------
# render — empty transcript
# ---------------------------------------------------------------------------

class TestRenderEmpty:
    def test_render_produces_string(self):
        t = make_transcript()
        output = t.render()
        assert isinstance(output, str)

    def test_render_includes_heading(self):
        t = make_transcript()
        assert "# Meeting Transcript" in t.render()

    def test_render_empty_shows_dash_for_speakers(self):
        t = make_transcript()
        assert "**Speakers:** —" in t.render()

    def test_render_empty_shows_unknown_duration(self):
        t = make_transcript()
        assert "Duration:** unknown" in t.render()

    def test_render_includes_source(self):
        t = make_transcript(source="Teams")
        assert "Teams" in t.render()

    def test_render_includes_date(self):
        t = make_transcript()
        assert "2024-03-15" in t.render()


# ---------------------------------------------------------------------------
# render — with entries
# ---------------------------------------------------------------------------

class TestRenderWithEntries:
    def test_render_includes_speaker_names(self, sample_transcript):
        output = sample_transcript.render()
        assert "Alice" in output
        assert "Bob" in output

    def test_render_includes_text(self, sample_transcript):
        output = sample_transcript.render()
        assert "Hello everyone." in output
        assert "Hi there." in output

    def test_render_timestamp_format(self):
        t = make_transcript()
        t.add_entry(3661.0, "Alice", "Late entry.")  # 1h 1m 1s
        output = t.render()
        assert "01:01:01" in output

    def test_render_consecutive_same_speaker_merged(self):
        """Two consecutive entries from the same speaker share one header line."""
        t = make_transcript()
        t.add_entry(0.0, "Alice", "Line one.")
        t.add_entry(2.0, "Alice", "Line two.")
        output = t.render()
        # Only one header for Alice
        assert output.count("**[00:00:00] Alice:**") == 1

    def test_render_speakers_list_in_order_of_appearance(self):
        t = make_transcript()
        t.add_entry(0.0, "Charlie", "Hi.")
        t.add_entry(1.0, "Alice", "Hey.")
        output = t.render()
        speakers_line = [l for l in output.splitlines() if "**Speakers:**" in l][0]
        assert speakers_line.index("Charlie") < speakers_line.index("Alice")

    def test_render_includes_footer(self):
        t = make_transcript()
        assert "faster-whisper" in t.render()


# ---------------------------------------------------------------------------
# render — duration formatting
# ---------------------------------------------------------------------------

class TestDurationFormatting:
    def _transcript_with_duration(self, minutes: int) -> MarkdownTranscript:
        start = datetime(2024, 1, 1, 8, 0, 0)
        t = MarkdownTranscript(source="Test", start_time=start)
        t.set_end_time(start + timedelta(minutes=minutes))
        return t

    def test_duration_minutes_only(self):
        t = self._transcript_with_duration(45)
        assert "45 minutes" in t.render()

    def test_duration_one_minute_singular(self):
        t = self._transcript_with_duration(1)
        assert "1 minute" in t.render()
        assert "minutes" not in t.render()

    def test_duration_hours_and_minutes(self):
        t = self._transcript_with_duration(90)
        output = t.render()
        assert "1 hour" in output
        assert "30 minutes" in output

    def test_duration_exact_hour(self):
        t = self._transcript_with_duration(60)
        assert "1 hour" in t.render()

    def test_duration_two_hours_exact(self):
        t = self._transcript_with_duration(120)
        assert "2 hours" in t.render()
