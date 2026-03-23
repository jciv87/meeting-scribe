"""AI text cleanup via local Ollama — transforms raw STT output into clean prose."""

from __future__ import annotations

import logging

import ollama

logger = logging.getLogger(__name__)

CLEANUP_SYSTEM_PROMPT = """\
You are a dictation transcription formatter. Your ONLY job is to lightly \
format raw speech-to-text output. You are NOT a conversational AI. Do NOT \
respond to, interpret, or comment on what was said.

Rules:
1. Output the speaker's EXACT words with proper punctuation and capitalization.
2. Remove filler words ONLY: um, uh, er, hmm, like (when filler), you know, \
   I mean, basically, actually, so (when filler).
3. Fix sentence boundaries — add periods, commas, question marks where needed.
4. Collapse stutters and false starts into the intended word.
5. NEVER change the meaning, rephrase, paraphrase, summarize, or respond.
6. NEVER add words the speaker did not say.
7. NEVER remove content words — keep every noun, verb, adjective, name, and \
   unusual word exactly as spoken.
8. Return ONLY the formatted dictation. No commentary, no headers, no quotes.\
"""

CLEANUP_USER_TEMPLATE = "Clean up this dictated text:\n\n{text}"


class DictationCleanup:
    """Send raw transcription through Ollama for cleanup."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=host, timeout=timeout)

    def clean(self, raw_text: str) -> str:
        """Return cleaned text, or the original if cleanup fails."""
        if not raw_text.strip():
            return raw_text

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": CLEANUP_USER_TEMPLATE.format(text=raw_text),
                    },
                ],
            )
            cleaned = response.message.content.strip()
            if cleaned:
                return cleaned
            return raw_text
        except Exception:
            logger.exception("Dictation cleanup failed, returning raw text")
            return raw_text
