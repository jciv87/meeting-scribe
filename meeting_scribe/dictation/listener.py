"""Push-to-talk and toggle hotkey listener for dictation.

Supports two modes:
- push_to_hold: Hold Ctrl+Right Shift to record, release to stop + process.
- toggle: Tap Ctrl+Right Shift to start, tap again to stop + process.

Uses pynput.keyboard.Listener (not GlobalHotKeys) to avoid the injected-arg
crash on macOS with Python 3.13.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

try:
    from pynput import keyboard as _keyboard

    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Map config strings to pynput Key members
_KEY_MAP: dict[str, object] = {}
if _PYNPUT_AVAILABLE:
    _KEY_MAP = {
        "shift_r": _keyboard.Key.shift_r,
        "shift_l": _keyboard.Key.shift_l,
        "alt_r": _keyboard.Key.alt_r,
        "alt_l": _keyboard.Key.alt_l,
        "ctrl": _keyboard.Key.ctrl,
        "ctrl_l": _keyboard.Key.ctrl_l,
        "ctrl_r": _keyboard.Key.ctrl_r,
        "cmd_r": _keyboard.Key.cmd_r,
        "cmd_l": _keyboard.Key.cmd_l,
    }


def _parse_hotkey(combo: str) -> tuple[set[object], object]:
    """Parse a hotkey string like 'ctrl+shift_r' into (modifiers, trigger).

    The last key in the combo is the trigger; all preceding keys are modifiers.
    Returns (modifier_set, trigger_key).
    """
    parts = [p.strip().lower() for p in combo.split("+")]
    if not parts:
        raise ValueError(f"Empty hotkey combo: {combo!r}")

    resolved = []
    for part in parts:
        if part not in _KEY_MAP:
            raise ValueError(
                f"Unknown key {part!r} in combo {combo!r}. "
                f"Available: {sorted(_KEY_MAP)}"
            )
        resolved.append(_KEY_MAP[part])

    if len(resolved) == 1:
        return set(), resolved[0]
    return set(resolved[:-1]), resolved[-1]


def _normalize(key: object) -> object:
    """Collapse left/right variants of ctrl so either side activates the combo."""
    if not _PYNPUT_AVAILABLE:
        return key
    if key in (_keyboard.Key.ctrl, _keyboard.Key.ctrl_l, _keyboard.Key.ctrl_r):
        return _keyboard.Key.ctrl
    return key


class DictationListener:
    """Listens for a hotkey combo and fires on_start / on_stop callbacks.

    Parameters
    ----------
    on_start : callable
        Called (in a new daemon thread) when recording should begin.
    on_stop : callable
        Called (in a new daemon thread) when recording should stop.
    hotkey : str
        Key combo string, e.g. ``"ctrl+shift_r"``.
    mode : str
        ``"push_to_hold"`` or ``"toggle"``.
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        hotkey: str = "ctrl+shift_r",
        mode: str = "push_to_hold",
    ) -> None:
        self._on_start = on_start
        self._on_stop = on_stop
        self._mode = mode

        self._modifiers: set[object] = set()
        self._trigger: object = None
        if _PYNPUT_AVAILABLE:
            self._modifiers, self._trigger = _parse_hotkey(hotkey)

        self._pressed: set[object] = set()
        self._recording = False
        self._lock = threading.Lock()
        self._listener: object | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not _PYNPUT_AVAILABLE:
            logger.warning("pynput not available — dictation hotkey disabled")
            return
        try:
            self._listener = _keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()  # type: ignore[attr-defined]
            logger.info(
                "Dictation hotkey active (mode=%s, trigger=%s)",
                self._mode,
                self._trigger,
            )
        except Exception:
            logger.exception("Failed to start dictation hotkey listener")
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._listener = None

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _combo_active(self) -> bool:
        """Return True when all required modifiers are held."""
        for mod in self._modifiers:
            if _normalize(mod) not in self._pressed:
                return False
        return True

    def _on_press(self, key: object) -> None:
        nk = _normalize(key)
        self._pressed.add(nk)

        # Did the trigger key just fire while modifiers are held?
        if key == self._trigger and self._combo_active():
            if self._mode == "push_to_hold":
                self._begin_recording()
            else:  # toggle
                # Read state atomically, then call begin/end (which lock internally)
                with self._lock:
                    currently_recording = self._recording
                if currently_recording:
                    self._end_recording()
                else:
                    self._begin_recording()

    def _on_release(self, key: object) -> None:
        nk = _normalize(key)
        self._pressed.discard(nk)

        if self._mode == "push_to_hold" and key == self._trigger:
            self._end_recording()

    def _begin_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True
        threading.Thread(target=self._on_start, daemon=True).start()

    def _end_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
        threading.Thread(target=self._on_stop, daemon=True).start()
