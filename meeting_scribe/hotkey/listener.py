"""Global hotkey listener using pynput.

Uses keyboard.Listener (not GlobalHotKeys) to avoid a pynput 1.8.x bug on
macOS where GlobalHotKeys._on_press() crashes with:
    TypeError: _on_press() missing 1 required positional argument: 'injected'
The bug affects pynput 1.7+/1.8.x on macOS with Python 3.13. keyboard.Listener
does not exhibit this issue because the injected argument is consumed internally
before reaching the caller-supplied on_press callback.
"""

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
        self._listener: object | None = None

    def start(self) -> None:
        """Start listening for the global hotkey."""
        if not _PYNPUT_AVAILABLE:
            return

        def _on_press(key: object) -> None:
            # Fn+F12 arrives as plain Key.f12 at the OS level on macOS.
            if key == _keyboard.Key.f12:
                try:
                    self.on_toggle()
                except Exception:
                    pass

        try:
            self._listener = _keyboard.Listener(on_press=_on_press)
            self._listener.start()  # type: ignore[attr-defined]
        except Exception:
            self._listener = None

    def stop(self) -> None:
        """Stop listening for the global hotkey."""
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._listener = None
