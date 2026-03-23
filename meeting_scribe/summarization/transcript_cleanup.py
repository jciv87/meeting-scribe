"""Transcript cleanup engine — polishes raw meeting transcripts via Ollama.

Produces a cleaned version of the full transcript with fillers removed,
grammar fixed, and punctuation corrected — while preserving speaker labels,
timestamps, and the original document structure.

Output is saved as ``{stem}_cleaned.md`` alongside the raw transcript.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

CLEANUP_SYSTEM_PROMPT = """\
You are a transcript formatter. You receive a raw meeting transcript and \
return a cleaned version. You are NOT a conversational AI. Do NOT respond \
to, summarize, or comment on the content.

Rules:
1. Preserve ALL structure exactly: headings, metadata block, speaker labels, \
   timestamps, and the horizontal rules.
2. For each speaker's text: fix punctuation, capitalization, and grammar. \
   Remove filler words (um, uh, er, hmm, like, you know, I mean, basically, \
   actually). Collapse stutters and false starts.
3. NEVER change the meaning, rephrase ideas, add words not spoken, or remove \
   content words.
4. NEVER remove or reorder speakers or timestamps.
5. NEVER add commentary, summaries, or notes.
6. Keep unusual or technical words exactly as spoken.
7. Return the FULL cleaned transcript in the same Markdown format.\
"""

CLEANUP_USER_TEMPLATE = "Clean up this meeting transcript:\n\n{transcript}"

# Max chars per chunk to stay within context window (~6k tokens ≈ ~24k chars)
_MAX_CHUNK_CHARS = 20_000


class TranscriptCleanupEngine:
    """Clean up a raw meeting transcript via Ollama."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    def clean(self, transcript_text: str) -> str:
        """Clean a transcript string, chunking if necessary."""
        if len(transcript_text) <= _MAX_CHUNK_CHARS:
            return self._clean_chunk(transcript_text)

        # Split on speaker entries (lines starting with **[timestamp] Speaker:**)
        header, body = self._split_header(transcript_text)
        chunks = self._chunk_body(body)

        cleaned_parts = [header] if header else []
        for chunk in chunks:
            cleaned_parts.append(self._clean_chunk(chunk))

        return "\n\n".join(cleaned_parts)

    def clean_file(self, transcript_path: Path) -> Path:
        """Read a transcript, clean it, and write alongside as _cleaned.md."""
        transcript_text = transcript_path.read_text(encoding="utf-8")
        cleaned = self.clean(transcript_text)

        cleaned_path = transcript_path.with_name(
            transcript_path.stem + "_cleaned.md"
        )
        cleaned_path.write_text(cleaned, encoding="utf-8")
        logger.info("Cleaned transcript saved to %s", cleaned_path)
        return cleaned_path

    def _clean_chunk(self, text: str) -> str:
        """Send a single chunk to Ollama for cleanup."""
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CLEANUP_USER_TEMPLATE.format(transcript=text),
                },
            ],
        )
        return response.message.content.strip()

    @staticmethod
    def _split_header(text: str) -> tuple[str, str]:
        """Split transcript into header (metadata) and body (entries).

        The header ends at the first ``---`` separator followed by
        ``## Transcript`` or the first speaker entry.
        """
        match = re.search(r"^## Transcript\s*$", text, re.MULTILINE)
        if match:
            return text[: match.end()].strip(), text[match.end() :].strip()

        match = re.search(r"^\*\*\[\d{2}:\d{2}:\d{2}\]", text, re.MULTILINE)
        if match:
            return text[: match.start()].strip(), text[match.start() :].strip()

        return "", text

    @staticmethod
    def _chunk_body(body: str) -> list[str]:
        """Split transcript body into chunks that fit within context window."""
        entries = re.split(r"(?=^\*\*\[\d{2}:\d{2}:\d{2}\])", body, flags=re.MULTILINE)
        entries = [e for e in entries if e.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for entry in entries:
            entry_len = len(entry)
            if current_len + entry_len > _MAX_CHUNK_CHARS and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(entry)
            current_len += entry_len

        if current:
            chunks.append("\n".join(current))

        return chunks
