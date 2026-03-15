#!/usr/bin/env python3
"""Dictation-to-text: speak and have text typed into the focused app.

Uses faster-whisper via RealtimeSTT for VAD-triggered transcription.
Text is injected via clipboard paste (pbcopy + Cmd-V).

Usage:
    python dictation.py              # Start with defaults
    python dictation.py --model base.en  # Use a smaller model for speed

Prerequisites:
    brew install portaudio
    pip install RealtimeSTT pyperclip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading

from pynput import keyboard


def type_text(text: str) -> None:
    """Inject text into the focused app via clipboard paste."""
    if not text.strip():
        return
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local voice dictation via faster-whisper")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--compute-type", default="int8", help="Compute type")
    args = parser.parse_args()

    try:
        from RealtimeSTT import AudioToTextRecorder
    except ImportError:
        print("RealtimeSTT not installed. Run: pip install RealtimeSTT", file=sys.stderr)
        sys.exit(1)

    # State
    active = threading.Event()
    active.set()  # Start active by default

    def on_transcription(text: str) -> None:
        if active.is_set() and text.strip():
            type_text(text + " ")

    print(f"Loading model: {args.model} ({args.compute_type})...")

    recorder = AudioToTextRecorder(
        model=args.model,
        compute_type=args.compute_type,
        language=args.language,
        spinner=False,
        silero_sensitivity=0.4,
        post_speech_silence_duration=0.6,
        on_realtime_transcription_stabilized=on_transcription,
    )

    # Global hotkey: Fn+F11 toggles dictation on/off
    def on_press(key: keyboard.Key) -> None:
        if key == keyboard.Key.f11:
            if active.is_set():
                active.clear()
                print("\n[Dictation paused]")
            else:
                active.set()
                print("\n[Dictation active]")

    hotkey_listener = keyboard.Listener(on_press=on_press)
    hotkey_listener.daemon = True
    hotkey_listener.start()

    print("Dictation active. Speak into your microphone.")
    print("Press F11 to toggle on/off. Ctrl-C to quit.")
    print()

    try:
        while True:
            recorder.text(on_transcription)
    except KeyboardInterrupt:
        print("\nDictation stopped.")


if __name__ == "__main__":
    main()
