"""Markdown transcript renderer for Meeting Scribe."""

from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptEntry:
    timestamp: float  # seconds relative to meeting start
    speaker: str
    text: str


class MarkdownTranscript:
    """Stores meeting metadata and transcript entries, renders to Markdown."""

    def __init__(self, source: str, start_time: datetime) -> None:
        self.source = source
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self._entries: list[TranscriptEntry] = []

    def add_entry(self, timestamp: float, speaker: str, text: str) -> None:
        """Append a transcript entry."""
        self._entries.append(TranscriptEntry(timestamp=timestamp, speaker=speaker, text=text))

    def set_end_time(self, end_time: datetime) -> None:
        self.end_time = end_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_timestamp(self, seconds: float) -> str:
        """Return HH:MM:SS string for a relative offset in seconds."""
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _format_clock(self, dt: datetime) -> str:
        """Return 12-hour clock string, e.g. '10:00 AM'."""
        return dt.strftime("%I:%M %p").lstrip("0")

    def _format_duration(self) -> str:
        """Return human-readable duration string."""
        if self.end_time is None:
            return "unknown"
        delta = self.end_time - self.start_time
        total_minutes = int(delta.total_seconds() // 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        if hours == 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        if minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return (
            f"{hours} hour{'s' if hours != 1 else ''} "
            f"{minutes} minute{'s' if minutes != 1 else ''}"
        )

    def _unique_speakers(self) -> list[str]:
        """Return speakers in order of first appearance, deduplicated."""
        seen: list[str] = []
        for entry in self._entries:
            if entry.speaker not in seen:
                seen.append(entry.speaker)
        return seen

    # ------------------------------------------------------------------
    # Public render
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Produce the full Markdown document."""
        date_str = self.start_time.strftime("%Y-%m-%d")
        start_clock = self._format_clock(self.start_time)
        end_clock = self._format_clock(self.end_time) if self.end_time else "—"
        duration = self._format_duration()
        speakers = ", ".join(self._unique_speakers()) if self._entries else "—"

        lines: list[str] = [
            "# Meeting Transcript",
            "",
            f"- **Date:** {date_str}",
            f"- **Time:** {start_clock} -- {end_clock}",
            f"- **Duration:** {duration}",
            f"- **Source:** {self.source}",
            f"- **Speakers:** {speakers}",
            "",
            "---",
            "",
            "## Transcript",
            "",
        ]

        # Merge consecutive entries from the same speaker.
        prev_speaker: Optional[str] = None
        for entry in self._entries:
            ts = self._format_timestamp(entry.timestamp)
            if entry.speaker != prev_speaker:
                if prev_speaker is not None:
                    lines.append("")
                lines.append(f"**[{ts}] {entry.speaker}:**")
                prev_speaker = entry.speaker
            lines.append(entry.text)

        lines += [
            "",
            "---",
            "",
            "*Transcribed by Meeting Scribe using faster-whisper*",
        ]

        return "\n".join(lines)
