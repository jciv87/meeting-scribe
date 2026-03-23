"""Tests for meeting_scribe.dictation.listener — hotkey parsing and state machine."""

import threading
import time
from unittest.mock import MagicMock

import pytest
from pynput import keyboard as _keyboard

from meeting_scribe.dictation.listener import (
    DictationListener,
    _normalize,
    _parse_hotkey,
)


# ---------------------------------------------------------------------------
# _parse_hotkey
# ---------------------------------------------------------------------------


class TestParseHotkey:
    def test_single_key(self):
        mods, trigger = _parse_hotkey("shift_r")
        assert mods == set()
        assert trigger == _keyboard.Key.shift_r

    def test_combo_ctrl_shift_r(self):
        mods, trigger = _parse_hotkey("ctrl+shift_r")
        assert trigger == _keyboard.Key.shift_r
        assert _keyboard.Key.ctrl in mods

    def test_combo_with_spaces(self):
        mods, trigger = _parse_hotkey(" ctrl + shift_r ")
        assert trigger == _keyboard.Key.shift_r

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            _parse_hotkey("ctrl+f5")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            _parse_hotkey("")

    def test_three_key_combo(self):
        mods, trigger = _parse_hotkey("ctrl+alt_r+shift_r")
        assert trigger == _keyboard.Key.shift_r
        assert _keyboard.Key.ctrl in mods
        assert _keyboard.Key.alt_r in mods


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_ctrl_l_normalizes_to_ctrl(self):
        assert _normalize(_keyboard.Key.ctrl_l) == _keyboard.Key.ctrl

    def test_ctrl_r_normalizes_to_ctrl(self):
        assert _normalize(_keyboard.Key.ctrl_r) == _keyboard.Key.ctrl

    def test_ctrl_normalizes_to_ctrl(self):
        assert _normalize(_keyboard.Key.ctrl) == _keyboard.Key.ctrl

    def test_shift_r_passes_through(self):
        assert _normalize(_keyboard.Key.shift_r) == _keyboard.Key.shift_r

    def test_alt_r_passes_through(self):
        assert _normalize(_keyboard.Key.alt_r) == _keyboard.Key.alt_r


# ---------------------------------------------------------------------------
# DictationListener — push_to_hold mode
# ---------------------------------------------------------------------------


class TestPushToHold:
    def _make_listener(self):
        on_start = MagicMock()
        on_stop = MagicMock()
        dl = DictationListener(
            on_start=on_start,
            on_stop=on_stop,
            hotkey="ctrl+shift_r",
            mode="push_to_hold",
        )
        return dl, on_start, on_stop

    def test_press_combo_starts_recording(self):
        dl, on_start, on_stop = self._make_listener()
        # Simulate: press ctrl, then press shift_r
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()
        on_stop.assert_not_called()

    def test_release_trigger_stops_recording(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        dl._on_release(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()
        on_stop.assert_called_once()

    def test_trigger_without_modifier_does_nothing(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_not_called()

    def test_double_press_does_not_double_start(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()

    def test_release_without_press_does_nothing(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_release(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_stop.assert_not_called()

    def test_ctrl_r_also_works_as_modifier(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.ctrl_r)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()


# ---------------------------------------------------------------------------
# DictationListener — toggle mode
# ---------------------------------------------------------------------------


class TestToggleMode:
    def _make_listener(self):
        on_start = MagicMock()
        on_stop = MagicMock()
        dl = DictationListener(
            on_start=on_start,
            on_stop=on_stop,
            hotkey="ctrl+shift_r",
            mode="toggle",
        )
        return dl, on_start, on_stop

    def test_first_tap_starts(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()
        on_stop.assert_not_called()

    def test_second_tap_stops(self):
        dl, on_start, on_stop = self._make_listener()
        # First tap — start
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        dl._on_release(_keyboard.Key.shift_r)
        dl._on_release(_keyboard.Key.ctrl_l)
        # Second tap — stop
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_start.assert_called_once()
        on_stop.assert_called_once()

    def test_release_does_not_stop_in_toggle_mode(self):
        dl, on_start, on_stop = self._make_listener()
        dl._on_press(_keyboard.Key.ctrl_l)
        dl._on_press(_keyboard.Key.shift_r)
        time.sleep(0.05)
        dl._on_release(_keyboard.Key.shift_r)
        time.sleep(0.05)
        on_stop.assert_not_called()


# ---------------------------------------------------------------------------
# Single-key hotkey
# ---------------------------------------------------------------------------


class TestSingleKeyHotkey:
    def test_single_key_push_to_hold(self):
        on_start = MagicMock()
        on_stop = MagicMock()
        dl = DictationListener(
            on_start=on_start,
            on_stop=on_stop,
            hotkey="alt_r",
            mode="push_to_hold",
        )
        dl._on_press(_keyboard.Key.alt_r)
        time.sleep(0.05)
        on_start.assert_called_once()
        dl._on_release(_keyboard.Key.alt_r)
        time.sleep(0.05)
        on_stop.assert_called_once()
