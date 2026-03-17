"""Tests for meeting_scribe.health — all external deps mocked."""

import pytest
from unittest.mock import patch, MagicMock

from meeting_scribe.health import (
    check_audio_routing,
    check_ollama,
    run_startup_checks,
    _check_multi_output_device,
    _get_current_output_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_device_list(*names: str) -> list[dict]:
    return [{"name": n} for n in names]


# ---------------------------------------------------------------------------
# check_audio_routing
# ---------------------------------------------------------------------------

class TestCheckAudioRouting:
    def test_device_found_no_warnings_when_routing_ok(self):
        """BlackHole present, multi-output exists, system output is blackhole."""
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("BlackHole 2ch")),
            patch("meeting_scribe.health.sd.query_devices"),
            patch("meeting_scribe.health.sd.query_devices", side_effect=[
                make_device_list("BlackHole 2ch"),       # first call: list all
                {"name": "Multi-Output Device"},         # second call: output device
            ]),
            patch("meeting_scribe.health._check_multi_output_device", return_value=True),
            patch("meeting_scribe.health._get_current_output_device", return_value="Multi-Output Device"),
        ):
            warnings = check_audio_routing("BlackHole 2ch")
        assert warnings == []

    def test_device_not_found_returns_warning(self):
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("Speaker", "Mic")),
            patch("meeting_scribe.health._check_multi_output_device", return_value=False),
            patch("meeting_scribe.health._get_current_output_device", return_value="Speaker"),
        ):
            warnings = check_audio_routing("BlackHole 2ch")
        assert len(warnings) == 1
        assert "BlackHole 2ch" in warnings[0]

    def test_device_not_found_stops_early(self):
        """When device missing, should return immediately without multi-output check."""
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("Speaker")),
            patch("meeting_scribe.health._check_multi_output_device") as mock_multi,
        ):
            check_audio_routing("BlackHole 2ch")
        mock_multi.assert_not_called()

    def test_no_multi_output_device_warns(self):
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("BlackHole 2ch")),
            patch("meeting_scribe.health._check_multi_output_device", return_value=False),
            patch("meeting_scribe.health._get_current_output_device", return_value="BlackHole 2ch"),
        ):
            warnings = check_audio_routing("BlackHole 2ch")
        assert any("Multi-Output" in w for w in warnings)

    def test_wrong_system_output_warns(self):
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("BlackHole 2ch")),
            patch("meeting_scribe.health._check_multi_output_device", return_value=True),
            patch("meeting_scribe.health._get_current_output_device", return_value="MacBook Speakers"),
        ):
            warnings = check_audio_routing("BlackHole 2ch")
        assert any("MacBook Speakers" in w for w in warnings)

    def test_system_output_contains_blackhole_no_routing_warning(self):
        with (
            patch("meeting_scribe.health.sd.query_devices", return_value=make_device_list("BlackHole 2ch")),
            patch("meeting_scribe.health._check_multi_output_device", return_value=True),
            patch("meeting_scribe.health._get_current_output_device", return_value="BlackHole 2ch"),
        ):
            warnings = check_audio_routing("BlackHole 2ch")
        # Routing warning should not appear
        assert not any("Switch to" in w for w in warnings)


# ---------------------------------------------------------------------------
# _check_multi_output_device
# ---------------------------------------------------------------------------

class TestCheckMultiOutputDevice:
    def test_returns_true_when_multi_output_in_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "Multi-Output Device: BlackHole\n"
        with patch("meeting_scribe.health.subprocess.run", return_value=mock_result):
            assert _check_multi_output_device() is True

    def test_returns_false_when_not_present(self):
        mock_result = MagicMock()
        mock_result.stdout = "Speaker:\n  Some device\n"
        with patch("meeting_scribe.health.subprocess.run", return_value=mock_result):
            assert _check_multi_output_device() is False

    def test_returns_false_on_timeout(self):
        import subprocess
        with patch("meeting_scribe.health.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            assert _check_multi_output_device() is False

    def test_returns_false_on_file_not_found(self):
        with patch("meeting_scribe.health.subprocess.run", side_effect=FileNotFoundError):
            assert _check_multi_output_device() is False

    def test_returns_true_for_meeting_scribe_device(self):
        mock_result = MagicMock()
        mock_result.stdout = "meeting-scribe audio device\n"
        with patch("meeting_scribe.health.subprocess.run", return_value=mock_result):
            assert _check_multi_output_device() is True


# ---------------------------------------------------------------------------
# _get_current_output_device
# ---------------------------------------------------------------------------

class TestGetCurrentOutputDevice:
    def test_returns_device_name(self):
        with patch("meeting_scribe.health.sd.query_devices", return_value={"name": "MacBook Speakers"}):
            result = _get_current_output_device()
        assert result == "MacBook Speakers"

    def test_returns_none_on_exception(self):
        with patch("meeting_scribe.health.sd.query_devices", side_effect=Exception("no device")):
            result = _get_current_output_device()
        assert result is None

    def test_returns_none_when_result_not_dict(self):
        with patch("meeting_scribe.health.sd.query_devices", return_value=None):
            result = _get_current_output_device()
        assert result is None


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------

class TestCheckOllama:
    def _make_ollama_client(self, model_names: list[str]):
        mock_client = MagicMock()
        models_response = MagicMock()
        models_response.models = [MagicMock(model=n) for n in model_names]
        mock_client.list.return_value = models_response
        return mock_client

    def test_model_present_no_warnings(self):
        client = self._make_ollama_client(["llama3.1:8b", "mistral"])
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = client
        with patch("meeting_scribe.health.ollama", mock_ollama):
            warnings = check_ollama(model="llama3.1:8b")
        assert warnings == []

    def test_model_missing_returns_warning(self):
        client = self._make_ollama_client(["mistral"])
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = client
        with patch("meeting_scribe.health.ollama", mock_ollama):
            warnings = check_ollama(model="llama3.1:8b")
        assert len(warnings) == 1
        assert "llama3.1:8b" in warnings[0]

    def test_ollama_unreachable_returns_warning(self):
        mock_ollama = MagicMock()
        mock_ollama.Client.side_effect = Exception("connection refused")
        with patch("meeting_scribe.health.ollama", mock_ollama):
            warnings = check_ollama()
        assert len(warnings) == 1
        assert "not reachable" in warnings[0]

    def test_ollama_import_error_returns_warning(self):
        """If ollama package isn't installed (ollama is None), check_ollama returns a warning."""
        with patch("meeting_scribe.health.ollama", None):
            warnings = check_ollama()
        assert len(warnings) >= 1
        assert "not reachable" in warnings[0]


# ---------------------------------------------------------------------------
# run_startup_checks
# ---------------------------------------------------------------------------

class TestRunStartupChecks:
    def test_no_warnings_when_all_ok(self):
        with (
            patch("meeting_scribe.health.check_audio_routing", return_value=[]),
            patch("meeting_scribe.health.check_ollama", return_value=[]),
        ):
            warnings = run_startup_checks()
        assert warnings == []

    def test_audio_warnings_included(self):
        with (
            patch("meeting_scribe.health.check_audio_routing", return_value=["audio warn"]),
            patch("meeting_scribe.health.check_ollama", return_value=[]),
        ):
            warnings = run_startup_checks()
        assert "audio warn" in warnings

    def test_ollama_warnings_included_when_enabled(self):
        with (
            patch("meeting_scribe.health.check_audio_routing", return_value=[]),
            patch("meeting_scribe.health.check_ollama", return_value=["ollama warn"]),
        ):
            warnings = run_startup_checks(summarization_enabled=True)
        assert "ollama warn" in warnings

    def test_ollama_not_checked_when_disabled(self):
        with (
            patch("meeting_scribe.health.check_audio_routing", return_value=[]),
            patch("meeting_scribe.health.check_ollama") as mock_check,
        ):
            run_startup_checks(summarization_enabled=False)
        mock_check.assert_not_called()

    def test_all_warnings_combined(self):
        with (
            patch("meeting_scribe.health.check_audio_routing", return_value=["w1", "w2"]),
            patch("meeting_scribe.health.check_ollama", return_value=["w3"]),
        ):
            warnings = run_startup_checks(summarization_enabled=True)
        assert set(warnings) == {"w1", "w2", "w3"}
