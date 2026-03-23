"""Tests for meeting_scribe.dictation.cleanup — Ollama text cleanup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from meeting_scribe.dictation.cleanup import (
    CLEANUP_SYSTEM_PROMPT,
    CLEANUP_USER_TEMPLATE,
    DictationCleanup,
)


# ---------------------------------------------------------------------------
# DictationCleanup.clean
# ---------------------------------------------------------------------------


class TestDictationCleanup:
    def _make_cleanup(self, response_text: str = "Cleaned text.") -> DictationCleanup:
        cleanup = DictationCleanup.__new__(DictationCleanup)
        cleanup.model = "test-model"
        cleanup._client = MagicMock()
        cleanup._client.chat.return_value = SimpleNamespace(
            message=SimpleNamespace(content=response_text)
        )
        return cleanup

    def test_returns_cleaned_text(self):
        cleanup = self._make_cleanup("Hello, this is cleaned.")
        result = cleanup.clean("um hello uh this is like cleaned")
        assert result == "Hello, this is cleaned."

    def test_sends_correct_messages(self):
        cleanup = self._make_cleanup()
        cleanup.clean("raw input text")
        call_kwargs = cleanup._client.chat.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == CLEANUP_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "raw input text" in messages[1]["content"]

    def test_empty_input_returns_as_is(self):
        cleanup = self._make_cleanup()
        result = cleanup.clean("")
        assert result == ""
        cleanup._client.chat.assert_not_called()

    def test_whitespace_input_returns_as_is(self):
        cleanup = self._make_cleanup()
        result = cleanup.clean("   ")
        assert result == "   "
        cleanup._client.chat.assert_not_called()

    def test_empty_response_returns_original(self):
        cleanup = self._make_cleanup("")
        result = cleanup.clean("some raw text")
        assert result == "some raw text"

    def test_ollama_exception_returns_original(self):
        cleanup = self._make_cleanup()
        cleanup._client.chat.side_effect = ConnectionError("Ollama down")
        result = cleanup.clean("raw text that should survive")
        assert result == "raw text that should survive"

    def test_uses_configured_model(self):
        cleanup = self._make_cleanup()
        cleanup.clean("test")
        call_kwargs = cleanup._client.chat.call_args
        assert call_kwargs.kwargs["model"] == "test-model"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_user_template_includes_text(self):
        rendered = CLEANUP_USER_TEMPLATE.format(text="Hello world")
        assert "Hello world" in rendered

    def test_system_prompt_mentions_filler_words(self):
        assert "filler" in CLEANUP_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_punctuation(self):
        assert "punctuation" in CLEANUP_SYSTEM_PROMPT.lower()
