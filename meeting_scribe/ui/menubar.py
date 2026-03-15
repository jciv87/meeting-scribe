"""macOS menu bar app for Meeting Scribe."""

import os
import subprocess
from typing import Any

try:
    import rumps
    _RUMPS_AVAILABLE = True
except ImportError:
    _RUMPS_AVAILABLE = False


class MeetingScribeMenuBar:
    """Menu bar application that exposes recording controls and status."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self._app: Any = None

        if not _RUMPS_AVAILABLE:
            return

        self._app = rumps.App("MeetingScribe", title="MS", quit_button=None)
        self._recording_start: float | None = None

        # Menu items
        self._status_item = rumps.MenuItem("Idle")
        self._status_item.set_callback(None)

        self._toggle_item = rumps.MenuItem("Start Transcription", callback=self._on_toggle)

        open_folder_item = rumps.MenuItem(
            "Open Transcripts Folder", callback=self._on_open_folder
        )

        quit_item = rumps.MenuItem("Quit", callback=self._on_quit)

        self._app.menu = [
            self._status_item,
            rumps.separator,
            self._toggle_item,
            rumps.separator,
            open_folder_item,
            rumps.separator,
            quit_item,
        ]

        # Timer — fires every second to update status while recording
        self._timer = rumps.Timer(self._on_tick, 1)
        self._timer.start()

    def _on_tick(self, _sender: Any) -> None:
        if not _RUMPS_AVAILABLE:
            return
        if self.controller.is_recording:
            import time
            if self._recording_start is None:
                self._recording_start = time.monotonic()
            elapsed = int(time.monotonic() - self._recording_start)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            self._status_item.title = f"Recording ({h:02d}:{m:02d}:{s:02d})"
            self._toggle_item.title = "Stop Transcription"
        else:
            self._recording_start = None
            self._status_item.title = "Idle"
            self._toggle_item.title = "Start Transcription"

    def _on_toggle(self, _sender: Any) -> None:
        if self.controller.is_recording:
            self.controller.stop()
        else:
            self.controller.start()

    def _on_open_folder(self, _sender: Any) -> None:
        transcript_dir = os.path.expanduser("~/meeting-scribe/transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        subprocess.run(["open", transcript_dir], check=False)

    def _on_quit(self, _sender: Any) -> None:
        if self.controller.is_recording:
            self.controller.stop()
        if _RUMPS_AVAILABLE:
            rumps.quit_application()

    def run(self) -> None:
        """Start the rumps event loop (blocks until quit)."""
        if self._app is not None:
            self._app.run()
