"""Detection functions for active meetings on macOS."""

import subprocess


def detect_zoom() -> str | None:
    """Check for Zoom's CptHost process. Returns 'Zoom' if found, None otherwise."""
    try:
        result = subprocess.run(
            ["pgrep", "CptHost"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            return "Zoom"
    except Exception:
        pass
    return None


def detect_google_meet() -> str | None:
    """Check Chrome tabs for meet.google.com. Returns 'Google Meet' if found, None otherwise."""
    script = """
    tell application "System Events"
        if not (exists process "Google Chrome") then
            return ""
        end if
    end tell
    tell application "Google Chrome"
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t contains "meet.google.com" then
                    return "found"
                end if
            end repeat
        end repeat
    end tell
    return ""
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=3,
            text=True,
        )
        if result.returncode == 0 and "found" in result.stdout:
            return "Google Meet"
    except Exception:
        pass
    return None


def detect_slack_huddle() -> str | None:
    """Check for Slack Helper process with Audio. Returns 'Slack Huddle' if found, None otherwise."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "Slack Helper.*Audio"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            return "Slack Huddle"
    except Exception:
        pass
    return None


def detect_active_meeting() -> str | None:
    """Check all meeting sources. Returns the first detected meeting name, or None."""
    for detector in (detect_zoom, detect_google_meet, detect_slack_huddle):
        result = detector()
        if result is not None:
            return result
    return None
