"""Startup health checks for Meeting Scribe."""

from __future__ import annotations

import logging
import subprocess

import sounddevice as sd

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def check_audio_routing(device_name: str = "BlackHole 2ch") -> list[str]:
    """Check audio device availability and routing.

    Returns a list of warning messages (empty if everything looks good).
    """
    warnings: list[str] = []

    # Check if BlackHole device exists
    devices = sd.query_devices()
    device_names = [d["name"] for d in devices]
    blackhole_found = any(device_name.lower() in name.lower() for name in device_names)

    if not blackhole_found:
        warnings.append(
            f"Audio device '{device_name}' not found. "
            f"Install BlackHole: brew install blackhole-2ch (then reboot)."
        )
        return warnings  # No point checking further

    # Check if a multi-output device exists that includes BlackHole
    has_multi_output = _check_multi_output_device()
    if not has_multi_output:
        warnings.append(
            "No Multi-Output Device detected. Meeting audio won't be captured.\n"
            "  Setup: Audio MIDI Setup → '+' → Create Multi-Output Device\n"
            "  → check your speakers AND BlackHole 2ch.\n"
            "  Then set system output to that device before meetings."
        )

    # Check if system output is routed through BlackHole
    current_output = _get_current_output_device()
    if current_output and "multi" not in current_output.lower() and "blackhole" not in current_output.lower() and "meeting" not in current_output.lower():
        warnings.append(
            f"System output is '{current_output}', not routed through BlackHole.\n"
            "  Switch to your Multi-Output Device in System Settings → Sound → Output\n"
            "  before starting a meeting."
        )

    return warnings


def _check_multi_output_device() -> bool:
    """Check if a multi-output aggregate device exists via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()
        return "multi-output" in output or "meeting scribe" in output or "meeting-scribe" in output
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _get_current_output_device() -> str | None:
    """Get the current default output device name."""
    try:
        device_info = sd.query_devices(kind="output")
        return device_info.get("name") if isinstance(device_info, dict) else None
    except Exception:
        return None


def check_ollama(host: str = "http://localhost:11434", model: str = "llama3.1:8b") -> list[str]:
    """Check if Ollama is reachable and the model is available.

    Returns a list of warning messages (empty if everything looks good).
    """
    warnings: list[str] = []

    try:
        if ollama is None:
            raise ImportError("ollama package not installed")
        client = ollama.Client(host=host, timeout=5)
        models = client.list()
        model_names = [m.model for m in models.models] if models.models else []

        if not any(model in name for name in model_names):
            warnings.append(
                f"Ollama model '{model}' not found. "
                f"Pull it with: ollama pull {model}"
            )
    except Exception:
        warnings.append(
            f"Ollama not reachable at {host}. "
            "Summarization will be unavailable. Start it with: ollama serve"
        )

    return warnings


def run_startup_checks(
    device_name: str = "BlackHole 2ch",
    summarization_enabled: bool = True,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:8b",
) -> list[str]:
    """Run all startup health checks. Returns list of warnings."""
    warnings: list[str] = []

    warnings.extend(check_audio_routing(device_name))

    if summarization_enabled:
        warnings.extend(check_ollama(ollama_host, ollama_model))

    return warnings
