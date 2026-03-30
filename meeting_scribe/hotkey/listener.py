"""Global hotkey listener using pynput.

Uses the UnifiedHotkeyListener singleton to avoid creating multiple
pynput.keyboard.Listener instances, which crashes on macOS due to
concurrent TIS/TSM API access.

On macOS, the Fn key modifies the key at hardware level before the OS sees
it, so Fn+F12 arrives as a plain F12 keypress. We therefore listen for
Key.f12 directly rather than trying to combine Fn with anything.
"""

from typing import Callable

from meeting_scribe.hotkey.unified import UnifiedHotkeyListener

try:
    from pynput import keyboard as _keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False


class HotkeyListener:
    """Listens for F12 and calls on_toggle when it fires."""

    def __init__(
        self,
        on_toggle: Callable[[], None],
        combination: str = "<fn>+<f12>",
    ) -> None:
        self.on_toggle = on_toggle
        self.combination = combination

    def start(self) -> None:
        """Register with the unified listener — does not create its own Listener."""
        if not _PYNPUT_AVAILABLE:
            return

        def _on_press(key: object) -> None:
            # Fn+F12 arrives as plain Key.f12 at the OS level on macOS.
            if key == _keyboard.Key.f12:
                try:
                    self.on_toggle()
                except Exception:
                    pass

        unified = UnifiedHotkeyListener.get_instance()
        unified.register("meeting-toggle", on_press=_on_press)

    def stop(self) -> None:
        """No-op — lifecycle managed by UnifiedHotkeyListener."""
