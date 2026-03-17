"""Floating always-on-top recording indicator using native AppKit.

Displays a small pill-shaped overlay with a pulsing red dot and status text.
Requires PyObjC (available in the project venv). Gracefully no-ops if AppKit
is unavailable (e.g., running headless in CI).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import AppKit
    import objc
    from AppKit import (
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSFont,
        NSMakeRect,
        NSPanel,
        NSScreen,
        NSTextField,
        NSTimer,
        NSView,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSRunLoop, NSDefaultRunLoopMode

    _APPKIT_AVAILABLE = True
except ImportError:
    _APPKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# No-op fallback (always defined, used when AppKit is unavailable)
# ---------------------------------------------------------------------------


class _NoOpOverlay:
    """Silent no-op used when AppKit is not available."""

    def show(self) -> None:
        pass

    def hide(self) -> None:
        pass

    def set_status(self, text: str) -> None:
        pass


# ---------------------------------------------------------------------------
# AppKit implementation (only built when PyObjC is present)
# ---------------------------------------------------------------------------


def _build_appkit_overlay_class():  # noqa: C901
    """Construct and return the AppKit-backed RecordingOverlay class."""

    class _DotView(NSView):
        """Small circular view that pulses between full and dim opacity."""

        def initWithFrame_(self, frame):  # noqa: N802
            self = objc.super(_DotView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._bright = True
            self._timer = None
            return self

        def start(self):
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.8, self, b"_tick:", None, True
            )
            try:
                NSRunLoop.currentRunLoop().addTimer_forMode_(
                    self._timer, NSDefaultRunLoopMode
                )
            except Exception:
                pass

        def stop(self):
            if self._timer is not None:
                self._timer.invalidate()
                self._timer = None

        @objc.typedSelector(b"v@:@")
        def _tick_(self, timer):  # noqa: N802
            self._bright = not self._bright
            self.setNeedsDisplay_(True)

        def drawRect_(self, dirty_rect):  # noqa: N802
            alpha = 1.0 if self._bright else 0.3
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.95, 0.2, 0.2, alpha
            ).set()
            path = NSBezierPath.bezierPathWithOvalInRect_(self.bounds())
            path.fill()

    class _DraggableView(NSView):
        """Content view that makes the borderless window draggable."""

        def mouseDown_(self, event):  # noqa: N802
            self._drag_start = event.locationInWindow()

        def mouseDragged_(self, event):  # noqa: N802
            if not hasattr(self, "_drag_start"):
                return
            window = self.window()
            if window is None:
                return
            loc = event.locationInWindow()
            origin = window.frame().origin
            new_x = origin.x + loc.x - self._drag_start.x
            new_y = origin.y + loc.y - self._drag_start.y
            window.setFrameOrigin_(AppKit.NSMakePoint(new_x, new_y))

        def drawRect_(self, dirty_rect):  # noqa: N802
            # Semi-transparent dark pill background
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.08, 0.08, 0.08, 0.82
            ).set()
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                self.bounds(), 18.0, 18.0
            )
            path.fill()

    class _OverlayHelper(AppKit.NSObject):
        """NSObject subclass for main-thread dispatch of overlay operations."""

        def init(self):
            self = objc.super(_OverlayHelper, self).init()
            if self is None:
                return None
            self.overlay = None
            return self

        @objc.typedSelector(b"v@:@")
        def doShow_(self, sender):  # noqa: N802
            if self.overlay is not None:
                self.overlay._main_thread_show()

        @objc.typedSelector(b"v@:@")
        def doHide_(self, sender):  # noqa: N802
            if self.overlay is not None:
                self.overlay._main_thread_hide()

        @objc.typedSelector(b"v@:@")
        def doSetText_(self, sender):  # noqa: N802
            if self.overlay is not None:
                self.overlay._main_thread_set_text()

    class _AppKitOverlay:
        """Floating always-on-top recording indicator using native AppKit."""

        _WIDTH = 132
        _HEIGHT = 36
        _MARGIN_X = 24
        _MARGIN_Y = 32

        def __init__(self) -> None:
            self._panel = None
            self._label = None
            self._dot_view = None
            self._pending_text = "Recording"
            self._helper = _OverlayHelper.alloc().init()
            self._helper.overlay = self
            self._build()

        # ------------------------------------------------------------------
        # Public API (thread-safe via main thread dispatch)
        # ------------------------------------------------------------------

        def show(self) -> None:
            """Show the overlay. Thread-safe — dispatches to main thread."""
            if self._panel is None:
                return
            self._helper.performSelectorOnMainThread_withObject_waitUntilDone_(
                b"doShow:", None, False
            )

        def hide(self) -> None:
            """Hide the overlay. Thread-safe — dispatches to main thread."""
            if self._panel is None:
                return
            self._helper.performSelectorOnMainThread_withObject_waitUntilDone_(
                b"doHide:", None, False
            )

        def set_status(self, text: str) -> None:
            """Update the status text. Thread-safe."""
            if self._label is None:
                return
            self._pending_text = text
            self._helper.performSelectorOnMainThread_withObject_waitUntilDone_(
                b"doSetText:", None, False
            )

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

        def _build(self) -> None:
            try:
                screen = NSScreen.mainScreen()
                if screen is not None:
                    screen_frame = screen.visibleFrame()
                else:
                    screen_frame = NSMakeRect(0, 0, 1280, 800)

                x = (
                    screen_frame.origin.x
                    + screen_frame.size.width
                    - self._WIDTH
                    - self._MARGIN_X
                )
                y = screen_frame.origin.y + self._MARGIN_Y

                style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
                panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                    NSMakeRect(x, y, self._WIDTH, self._HEIGHT),
                    style,
                    NSBackingStoreBuffered,
                    False,
                )
                panel.setLevel_(AppKit.NSFloatingWindowLevel)
                panel.setOpaque_(False)
                panel.setBackgroundColor_(NSColor.clearColor())
                panel.setHasShadow_(False)
                panel.setCollectionBehavior_(
                    AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                    | AppKit.NSWindowCollectionBehaviorStationary
                    | AppKit.NSWindowCollectionBehaviorIgnoresCycle
                )
                panel.setIgnoresMouseEvents_(False)
                panel.setMovableByWindowBackground_(False)

                # Content view (draggable pill background)
                content = _DraggableView.alloc().initWithFrame_(
                    NSMakeRect(0, 0, self._WIDTH, self._HEIGHT)
                )
                panel.setContentView_(content)

                # Red pulsing dot
                dot_size = 10
                dot_x = 12
                dot_y = (self._HEIGHT - dot_size) / 2
                dot_view = _DotView.alloc().initWithFrame_(
                    NSMakeRect(dot_x, dot_y, dot_size, dot_size)
                )
                content.addSubview_(dot_view)
                self._dot_view = dot_view

                # Status label
                label_x = dot_x + dot_size + 7
                label_w = self._WIDTH - label_x - 10
                label_h = 20
                label_y = (self._HEIGHT - label_h) / 2
                label = NSTextField.alloc().initWithFrame_(
                    NSMakeRect(label_x, label_y, label_w, label_h)
                )
                label.setStringValue_("Recording")
                label.setEditable_(False)
                label.setSelectable_(False)
                label.setBordered_(False)
                label.setDrawsBackground_(False)
                label.setTextColor_(NSColor.whiteColor())
                label.setFont_(
                    NSFont.systemFontOfSize_weight_(13.0, AppKit.NSFontWeightMedium)
                )
                content.addSubview_(label)
                self._label = label

                self._panel = panel
            except Exception:
                logger.exception("RecordingOverlay: failed to build NSPanel")
                self._panel = None

        def _main_thread_show(self) -> None:
            try:
                if self._dot_view is not None:
                    self._dot_view.start()
                self._panel.orderFrontRegardless()
            except Exception:
                logger.debug("RecordingOverlay.show failed", exc_info=True)

        def _main_thread_hide(self) -> None:
            try:
                if self._dot_view is not None:
                    self._dot_view.stop()
                self._panel.orderOut_(None)
            except Exception:
                logger.debug("RecordingOverlay.hide failed", exc_info=True)

        def _main_thread_set_text(self) -> None:
            try:
                self._label.setStringValue_(self._pending_text)
            except Exception:
                logger.debug("RecordingOverlay.set_status failed", exc_info=True)

    return _AppKitOverlay


# ---------------------------------------------------------------------------
# Public export — select implementation based on availability
# ---------------------------------------------------------------------------

if _APPKIT_AVAILABLE:
    try:
        RecordingOverlay = _build_appkit_overlay_class()
    except Exception:
        logger.warning(
            "RecordingOverlay: AppKit class construction failed, using no-op fallback",
            exc_info=True,
        )
        RecordingOverlay = _NoOpOverlay  # type: ignore[misc,assignment]
else:
    RecordingOverlay = _NoOpOverlay  # type: ignore[misc,assignment]
