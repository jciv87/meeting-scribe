"""Unified global hotkey listener — single pynput.keyboard.Listener for all hotkeys.

macOS aborts if TIS/TSM APIs are called from multiple threads concurrently.
Using two separate pynput.keyboard.Listener instances causes exactly this.
This module provides a single listener that dispatches to multiple handlers.
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


class _HotkeyHandler:
    """A registered hotkey with its callbacks."""

    def __init__(self, name: str, on_press: Callable[[object], None],
                 on_release: Callable[[object], None] | None = None):
        self.name = name
        self.on_press = on_press
        self.on_release = on_release


class UnifiedHotkeyListener:
    """Single pynput listener that dispatches key events to registered handlers."""

    _instance: UnifiedHotkeyListener | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._handlers: list[_HotkeyHandler] = []
        self._listener: object | None = None

    @classmethod
    def get_instance(cls) -> UnifiedHotkeyListener:
        """Get or create the singleton listener."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, name: str, on_press: Callable[[object], None],
                 on_release: Callable[[object], None] | None = None) -> None:
        """Register a handler that receives all key events."""
        self._handlers.append(_HotkeyHandler(name, on_press, on_release))
        logger.info("Registered hotkey handler: %s", name)

    def start(self) -> None:
        """Start the single global listener."""
        if not _PYNPUT_AVAILABLE:
            logger.warning("pynput not available — hotkeys disabled")
            return
        if self._listener is not None:
            return  # already running

        try:
            self._listener = _keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()  # type: ignore[attr-defined]
            logger.info("Unified hotkey listener started with %d handlers",
                        len(self._handlers))
        except Exception:
            logger.exception("Failed to start unified hotkey listener")
            self._listener = None

    def stop(self) -> None:
        """Stop the listener."""
        if self._listener is not None:
            try:
                self._listener.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._listener = None

    def _on_press(self, key: object) -> None:
        for handler in self._handlers:
            try:
                handler.on_press(key)
            except Exception:
                logger.exception("Error in hotkey handler %s on_press", handler.name)

    def _on_release(self, key: object) -> None:
        for handler in self._handlers:
            if handler.on_release is not None:
                try:
                    handler.on_release(key)
                except Exception:
                    logger.exception("Error in hotkey handler %s on_release", handler.name)
