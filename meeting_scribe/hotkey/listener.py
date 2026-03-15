"""Global hotkey listener using pynput."""

from typing import Callable

try:
    from pynput import keyboard as _keyboard
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False


class HotkeyListener:
    """Listens for a global hotkey and calls on_toggle when it fires.

    On macOS, the Fn key modifies the key at hardware level before the OS sees
    it, so Fn+F12 arrives as a plain F12 keypress. We therefore listen for
    Key.f12 directly rather than trying to combine Fn with anything.
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        combination: str = "<fn>+<f12>",
    ) -> None:
        self.on_toggle = on_toggle
        self.combination = combination
        self._hotkeys: object | None = None

    def start(self) -> None:
        """Start listening for the global hotkey."""
        if not _PYNPUT_AVAILABLE:
            return

        def _fire():
            try:
                self.on_toggle()
            except Exception:
                pass

        # Fn+F12 arrives as plain F12 at the OS level on macOS
        mapping = {
            "<f12>": _fire,
        }

        try:
            self._hotkeys = _keyboard.GlobalHotKeys(mapping)
            self._hotkeys.start()  # type: ignore[attr-defined]
        except Exception:
            self._hotkeys = None

    def stop(self) -> None:
        """Stop listening for the global hotkey."""
        if self._hotkeys is not None:
            try:
                self._hotkeys.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._hotkeys = None
