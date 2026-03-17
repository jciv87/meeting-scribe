#!/usr/bin/env python3
"""Dictation-to-text: speak and have text typed into the focused app.

Self-contained VAD-based dictation using faster-whisper, sounddevice, and
pynput. No RealtimeSTT required.

Architecture:
  - Audio thread: captures chunks from default input device, computes RMS
    energy per chunk, accumulates speech into a buffer, triggers transcription
    when silence exceeds SILENCE_DURATION after a speech segment.
  - Main thread: runs the keyboard listener and sleeps.
  - F11 toggles dictation on/off.
  - Ctrl-C to quit.

Usage:
    python dictation.py
    python dictation.py --model large-v3-turbo --language en
    python dictation.py --energy-threshold 0.015 --silence-duration 0.8
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz — whisper native rate
CHUNK_DURATION = 0.05         # seconds per audio callback chunk (50 ms)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_LANGUAGE = "en"
DEFAULT_ENERGY_THRESHOLD = 0.01   # RMS — tune up if environment is noisy
DEFAULT_SILENCE_DURATION = 0.6    # seconds of silence before transcribing
DEFAULT_MIN_SPEECH_DURATION = 0.3  # seconds — ignore very short bursts


# ---------------------------------------------------------------------------
# Text injection
# ---------------------------------------------------------------------------

def type_text(text: str) -> None:
    """Inject text into the focused app via clipboard paste."""
    if not text.strip():
        return
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to keystroke "v" using command down',
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# VAD audio loop
# ---------------------------------------------------------------------------

class DictationEngine:
    """Captures audio, detects speech via energy VAD, transcribes on silence."""

    def __init__(
        self,
        model: WhisperModel,
        language: str,
        energy_threshold: float,
        silence_duration: float,
        min_speech_duration: float,
    ) -> None:
        self._model = model
        self._language = language
        self._energy_threshold = energy_threshold
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration

        self._active = threading.Event()
        self._active.set()  # Start listening by default
        self._stop = threading.Event()

        # Speech accumulation state (accessed only from audio thread)
        self._speech_buffer: list[np.ndarray] = []
        self._silence_chunks: int = 0
        self._in_speech: bool = False

        # Transcription work queue so audio thread never blocks on whisper
        self._tx_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._tx_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._tx_thread.start()

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        if self._active.is_set():
            self._active.clear()
            # Flush any in-progress speech buffer on pause
            self._speech_buffer.clear()
            self._in_speech = False
            self._silence_chunks = 0
            print("\n[Paused]", flush=True)
        else:
            self._active.set()
            print("\n[Listening]", flush=True)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Audio stream
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the default input device and run the audio loop."""
        silence_chunk_threshold = int(
            self._silence_duration / CHUNK_DURATION
        )
        min_speech_chunks = int(
            self._min_speech_duration / CHUNK_DURATION
        )

        def _audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info,
            status,
        ) -> None:
            if status:
                # Non-fatal — log to stderr without cluttering stdout
                print(f"[audio] {status}", file=sys.stderr, flush=True)

            if not self._active.is_set():
                return

            samples = indata[:, 0].astype(np.float32)  # mono
            rms = float(np.sqrt(np.mean(samples ** 2)))
            is_speech = rms >= self._energy_threshold

            if is_speech:
                self._speech_buffer.append(samples.copy())
                self._silence_chunks = 0
                if not self._in_speech:
                    self._in_speech = True
            else:
                if self._in_speech:
                    self._silence_chunks += 1
                    # Keep buffering during the silence tail so we don't clip
                    self._speech_buffer.append(samples.copy())

                    if self._silence_chunks >= silence_chunk_threshold:
                        # Enough silence — fire transcription if segment long enough
                        if len(self._speech_buffer) >= min_speech_chunks:
                            audio = np.concatenate(self._speech_buffer)
                            self._tx_queue.put(audio)
                        # Reset state
                        self._speech_buffer.clear()
                        self._silence_chunks = 0
                        self._in_speech = False

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=_audio_callback,
        ):
            print("[Listening]", flush=True)
            while not self._stop.is_set():
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Transcription worker
    # ------------------------------------------------------------------

    def _transcription_worker(self) -> None:
        while True:
            try:
                audio = self._tx_queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue

            print("[Transcribing...]", end=" ", flush=True)
            try:
                segments, _ = self._model.transcribe(
                    audio,
                    language=self._language,
                    beam_size=5,
                    vad_filter=False,  # We do our own VAD
                    word_timestamps=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    temperature=0.0,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
            except Exception as exc:
                print(f"\n[Transcription error: {exc}]", file=sys.stderr, flush=True)
                self._tx_queue.task_done()
                continue

            if text:
                print(f'"{text}"', flush=True)
                type_text(text + " ")
            else:
                print("(no speech)", flush=True)

            self._tx_queue.task_done()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local voice dictation via faster-whisper (no RealtimeSTT)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help=f"Compute type for faster-whisper (default: {DEFAULT_COMPUTE_TYPE})",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help=f"RMS energy threshold for VAD (default: {DEFAULT_ENERGY_THRESHOLD})",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=DEFAULT_SILENCE_DURATION,
        help=f"Seconds of silence before transcribing (default: {DEFAULT_SILENCE_DURATION})",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=DEFAULT_MIN_SPEECH_DURATION,
        help=f"Minimum speech segment length in seconds (default: {DEFAULT_MIN_SPEECH_DURATION})",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model} ({args.compute_type})...", flush=True)
    model = WhisperModel(
        args.model,
        compute_type=args.compute_type,
        cpu_threads=4,
    )
    print("Model loaded.", flush=True)

    engine = DictationEngine(
        model=model,
        language=args.language,
        energy_threshold=args.energy_threshold,
        silence_duration=args.silence_duration,
        min_speech_duration=args.min_speech_duration,
    )

    def on_press(key: keyboard.Key) -> None:
        if key == keyboard.Key.f11:
            engine.toggle()

    hotkey_listener = keyboard.Listener(on_press=on_press)
    hotkey_listener.daemon = True
    hotkey_listener.start()

    print("Dictation active. Speak into your microphone.")
    print("Press F11 to toggle on/off. Ctrl-C to quit.")
    print(f"VAD threshold: {args.energy_threshold} RMS  |  "
          f"silence: {args.silence_duration}s  |  "
          f"min speech: {args.min_speech_duration}s")
    print()

    try:
        engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\nDictation stopped.")


if __name__ == "__main__":
    main()
