"""Meeting monitor that polls for active meetings and fires debounced callbacks."""

import threading
import time
from typing import Callable

from meeting_scribe.detection.apps import detect_active_meeting


class MeetingMonitor:
    """Polls for active meetings and fires callbacks with start/end debouncing."""

    def __init__(
        self,
        poll_interval: int = 5,
        start_debounce: int = 3,
        end_debounce: int = 10,
        on_meeting_start: Callable[[str], None] | None = None,
        on_meeting_end: Callable[[str], None] | None = None,
    ) -> None:
        self.poll_interval = poll_interval
        self.start_debounce = start_debounce
        self.end_debounce = end_debounce
        self.on_meeting_start = on_meeting_start
        self.on_meeting_end = on_meeting_end

        self.current_meeting: str | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Tracks how long a candidate state has been continuously observed
        self._candidate: str | None = None
        self._candidate_since: float = 0.0
        self._absent_since: float = 0.0

    def start(self) -> None:
        """Start the polling loop in a daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MeetingMonitor")
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 1)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(timeout=self.poll_interval)

    def _tick(self) -> None:
        detected = detect_active_meeting()
        now = time.monotonic()

        if self.current_meeting is None:
            # Looking for a meeting to start
            if detected is not None:
                if self._candidate != detected:
                    # New candidate — reset the clock
                    self._candidate = detected
                    self._candidate_since = now
                elif (now - self._candidate_since) >= self.start_debounce:
                    # Debounce satisfied — meeting has started
                    self.current_meeting = detected
                    self._candidate = None
                    if self.on_meeting_start is not None:
                        try:
                            self.on_meeting_start(detected)
                        except Exception:
                            pass
            else:
                # Nothing detected — reset candidate
                self._candidate = None
        else:
            # A meeting is in progress — watch for it to end
            if detected is None:
                if self._absent_since == 0.0:
                    self._absent_since = now
                elif (now - self._absent_since) >= self.end_debounce:
                    # Debounce satisfied — meeting has ended
                    ended_meeting = self.current_meeting
                    self.current_meeting = None
                    self._absent_since = 0.0
                    if self.on_meeting_end is not None:
                        try:
                            self.on_meeting_end(ended_meeting)
                        except Exception:
                            pass
            else:
                # Still in a meeting (possibly a different one — update name)
                self.current_meeting = detected
                self._absent_since = 0.0
