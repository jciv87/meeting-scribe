"""Summarization engine using a local LLM via Ollama."""

from __future__ import annotations

import logging
from pathlib import Path

import ollama

from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class SummarizationEngine:
    """Generate structured meeting summaries from transcript text."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    def summarize(self, transcript_text: str) -> str:
        """Send transcript to the local LLM and return a Markdown summary."""
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(transcript=transcript_text),
                },
            ],
        )
        return response.message.content

    def summarize_file(self, transcript_path: Path) -> Path:
        """Read a transcript file, summarize it, and write the summary alongside it.

        Returns the path to the summary file.
        """
        transcript_text = transcript_path.read_text(encoding="utf-8")
        summary = self.summarize(transcript_text)

        summary_path = transcript_path.with_name(
            transcript_path.stem + "_summary.md"
        )
        summary_path.write_text(summary, encoding="utf-8")
        logger.info("Summary saved to %s", summary_path)
        return summary_path
