"""Dictation pipeline: mic → buffer → faster-whisper → Ollama cleanup → paste.

Records from the system default input mic (not BlackHole), transcribes the
complete utterance with faster-whisper, cleans it up through Ollama, then
pastes the result into the frontmost application via osascript.
"""

from __future__ import annotations

import logging
import subprocess
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class DictationEngine:
    """Records audio, transcribes, cleans up, and pastes text."""

    def __init__(
        self,
        transcription_engine: object,
        cleanup: object,
        audio_device: str = "",
        sample_rate: int = 16000,
    ) -> None:
        from meeting_scribe.dictation.cleanup import DictationCleanup
        from meeting_scribe.transcription.engine import TranscriptionEngine

        self._engine: TranscriptionEngine = transcription_engine  # type: ignore[assignment]
        self._cleanup: DictationCleanup = cleanup  # type: ignore[assignment]
        self._sample_rate = sample_rate
        self._audio_device = audio_device

        self._recording = False
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> int | None:
        """Resolve the input device index. Returns None for system default."""
        if not self._audio_device:
            return None
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if (
                self._audio_device.lower() in dev["name"].lower()
                and dev["max_input_channels"] > 0
            ):
                return idx
        logger.warning(
            "Dictation audio device %r not found, using system default",
            self._audio_device,
        )
        return None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Open the mic stream and start buffering audio."""
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._chunks = []

        device_idx = self._resolve_device()

        try:
            self._stream = sd.InputStream(
                device=device_idx,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Dictation recording started")
        except Exception:
            logger.exception("Failed to open mic for dictation")
            with self._lock:
                self._recording = False

    def stop_recording(self) -> None:
        """Stop the mic, transcribe, clean up, and paste."""
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        # Stop stream
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Grab buffered audio
        with self._lock:
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            logger.info("Dictation stopped — no audio captured")
            return

        audio = np.concatenate(chunks)
        duration = len(audio) / self._sample_rate
        logger.info("Dictation captured %.1fs of audio", duration)

        if duration < 0.3:
            logger.info("Audio too short (%.1fs), skipping", duration)
            return

        # Process in background to avoid blocking the hotkey listener
        threading.Thread(
            target=self._process,
            args=(audio,),
            daemon=True,
            name="DictationProcess",
        ).start()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """Sounddevice callback — buffer incoming audio."""
        if not self._recording:
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        with self._lock:
            self._chunks.append(mono.copy())

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def _process(self, audio: np.ndarray) -> None:
        """Transcribe → cleanup → paste."""
        try:
            # 1. Transcribe
            raw_text = self._transcribe(audio)
            if not raw_text.strip():
                logger.info("Transcription returned empty text")
                self._notify("Dictation", "No speech detected")
                return

            logger.info("Raw transcription: %s", raw_text[:200])

            # 2. AI cleanup
            cleaned = self._cleanup.clean(raw_text)
            logger.info("Cleaned text: %s", cleaned[:200])

            # 3. Paste into active app
            self._paste(cleaned)
            self._notify("Dictation", f"Pasted {len(cleaned)} chars")

        except Exception:
            logger.exception("Dictation processing failed")
            self._notify("Dictation Error", "Processing failed — check logs")

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run faster-whisper on the audio buffer and return concatenated text."""
        segments = self._engine.transcribe_chunk(audio)
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(parts)

    @staticmethod
    def _paste(text: str) -> None:
        """Copy text to clipboard and paste into the frontmost app via osascript."""
        # Set the clipboard
        proc = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
        if proc.returncode != 0:
            logger.error("pbcopy failed: %s", proc.stderr.decode())
            return

        # Cmd+V in the frontmost app
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            capture_output=True,
            timeout=5,
        )
        logger.info("Text pasted via Cmd+V")

    @staticmethod
    def _notify(title: str, message: str) -> None:
        """Show a macOS notification (best-effort)."""
        try:
            import rumps

            rumps.notification(
                title="Meeting Scribe",
                subtitle=title,
                message=message,
            )
        except Exception:
            pass
