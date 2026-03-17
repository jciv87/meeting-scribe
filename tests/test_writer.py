"""Tests for meeting_scribe.output.writer.TranscriptWriter."""

import pytest
from datetime import datetime
from pathlib import Path

from meeting_scribe.output.markdown import MarkdownTranscript
from meeting_scribe.output.writer import TranscriptWriter


def make_writer(output_dir: Path, source: str = "Zoom") -> TranscriptWriter:
    start = datetime(2024, 6, 15, 10, 0, 0)
    t = MarkdownTranscript(source=source, start_time=start)
    t.add_entry(0.0, "Alice", "Hello.")
    return TranscriptWriter(output_dir=output_dir, transcript=t)


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------

class TestGetOutputPath:
    def test_output_path_is_markdown_file(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        assert w.get_output_path().suffix == ".md"

    def test_output_path_contains_date(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        assert "2024-06-15" in w.get_output_path().name

    def test_output_path_contains_source_slug(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        assert "zoom" in w.get_output_path().name

    def test_output_path_inside_output_dir(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        assert w.get_output_path().parent == tmp_transcript_dir

    def test_output_dir_created_if_missing(self, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        w = make_writer(subdir)
        assert subdir.exists()


# ---------------------------------------------------------------------------
# Filename collision avoidance
# ---------------------------------------------------------------------------

class TestFilenameCollision:
    def test_second_writer_gets_different_path(self, tmp_transcript_dir):
        w1 = make_writer(tmp_transcript_dir)
        w1.save_final()  # creates the file
        w2 = make_writer(tmp_transcript_dir)
        assert w1.get_output_path() != w2.get_output_path()

    def test_collision_suffix_is_incremented(self, tmp_transcript_dir):
        w1 = make_writer(tmp_transcript_dir)
        w1.save_final()
        w2 = make_writer(tmp_transcript_dir)
        assert "_2" in w2.get_output_path().name

    def test_triple_collision_reaches_3(self, tmp_transcript_dir):
        w1 = make_writer(tmp_transcript_dir)
        w1.save_final()
        w2 = make_writer(tmp_transcript_dir)
        w2.save_final()
        w3 = make_writer(tmp_transcript_dir)
        assert "_3" in w3.get_output_path().name

    def test_source_special_chars_are_slugified(self, tmp_transcript_dir):
        """Source names with spaces/special chars become safe slugs."""
        w = make_writer(tmp_transcript_dir, source="Google Meet!!!")
        name = w.get_output_path().name
        # Should not contain raw spaces or exclamation marks
        assert " " not in name
        assert "!" not in name


# ---------------------------------------------------------------------------
# save_partial
# ---------------------------------------------------------------------------

class TestSavePartial:
    def test_save_partial_creates_partial_file(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_partial()
        partial = w.get_output_path().with_suffix("").with_suffix(".partial.md")
        assert partial.exists()

    def test_save_partial_contains_rendered_content(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_partial()
        partial = w.get_output_path().with_suffix("").with_suffix(".partial.md")
        content = partial.read_text(encoding="utf-8")
        assert "# Meeting Transcript" in content
        assert "Alice" in content

    def test_save_partial_does_not_create_final_file(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_partial()
        assert not w.get_output_path().exists()


# ---------------------------------------------------------------------------
# save_final
# ---------------------------------------------------------------------------

class TestSaveFinal:
    def test_save_final_creates_md_file(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_final()
        assert w.get_output_path().exists()

    def test_save_final_content_is_rendered_markdown(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_final()
        content = w.get_output_path().read_text(encoding="utf-8")
        assert "# Meeting Transcript" in content

    def test_save_final_removes_partial_file(self, tmp_transcript_dir):
        w = make_writer(tmp_transcript_dir)
        w.save_partial()
        partial = w.get_output_path().with_suffix("").with_suffix(".partial.md")
        assert partial.exists()
        w.save_final()
        assert not partial.exists()

    def test_save_final_without_prior_partial_is_ok(self, tmp_transcript_dir):
        """save_final should not raise if no partial file exists."""
        w = make_writer(tmp_transcript_dir)
        w.save_final()  # no partial to remove
        assert w.get_output_path().exists()
