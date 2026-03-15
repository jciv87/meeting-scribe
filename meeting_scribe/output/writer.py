"""Transcript file writer for Meeting Scribe."""

import re
from pathlib import Path
from typing import Union

from .markdown import MarkdownTranscript


class TranscriptWriter:
    """Writes MarkdownTranscript to disk, handling partial and final saves."""

    def __init__(self, output_dir: Union[str, Path], transcript: MarkdownTranscript) -> None:
        self.output_dir = Path(output_dir)
        self.transcript = transcript
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path: Path = self._resolve_output_path()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _base_filename(self) -> str:
        """Return base filename stem derived from date and source."""
        date_str = self.transcript.start_time.strftime("%Y-%m-%d")
        source_slug = re.sub(r"[^\w-]", "", self.transcript.source.lower().replace(" ", "-"))[:64]
        return f"{date_str}_{source_slug}"

    def _resolve_output_path(self) -> Path:
        """Return a non-colliding .md path, appending _2, _3, ... if needed."""
        base = self._base_filename()
        candidate = self.output_dir / f"{base}.md"
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            candidate = self.output_dir / f"{base}_{counter}.md"
            if not candidate.exists():
                return candidate
            counter += 1

    def get_output_path(self) -> Path:
        """Return the final (non-partial) output path."""
        return self._output_path

    def _partial_path(self) -> Path:
        """Return the .partial.md path corresponding to the output path."""
        return self._output_path.with_suffix("").with_suffix(".partial.md")

    # ------------------------------------------------------------------
    # Save operations
    # ------------------------------------------------------------------

    def save_partial(self) -> None:
        """Write current transcript state to a .partial.md file."""
        partial = self._partial_path()
        partial.write_text(self.transcript.render(), encoding="utf-8")

    def save_final(self) -> None:
        """Write final transcript to .md file and remove .partial.md if present."""
        self._output_path.write_text(self.transcript.render(), encoding="utf-8")
        partial = self._partial_path()
        if partial.exists():
            partial.unlink()
