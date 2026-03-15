#!/bin/bash
# Meeting Scribe — Audio Setup
# Run this script once to install BlackHole and configure audio routing.

set -e

echo "=== Meeting Scribe Audio Setup ==="
echo ""

# Step 1: Install BlackHole
if brew list blackhole-2ch &>/dev/null; then
    echo "[OK] BlackHole 2ch is installed"
else
    echo "[INSTALL] Installing BlackHole 2ch (requires sudo)..."
    brew install blackhole-2ch
    echo ""
    echo "[!] You MUST reboot after installing BlackHole."
    echo "    Run this script again after rebooting to complete setup."
    exit 0
fi

# Step 2: Check if BlackHole device is available
if system_profiler SPAudioDataType 2>/dev/null | grep -q "BlackHole"; then
    echo "[OK] BlackHole audio device detected"
else
    echo "[!] BlackHole device not found. You may need to reboot."
    exit 1
fi

echo ""
echo "=== Manual Step Required ==="
echo ""
echo "Create a Multi-Output Device so you can hear audio AND capture it:"
echo ""
echo "  1. Open 'Audio MIDI Setup' (Spotlight: Cmd+Space, type 'Audio MIDI Setup')"
echo "  2. Click the '+' button at bottom-left"
echo "  3. Select 'Create Multi-Output Device'"
echo "  4. Check BOTH:"
echo "     - Your speakers/headphones (e.g., 'MacBook Pro Speakers' or 'AirPods')"
echo "     - 'BlackHole 2ch'"
echo "  5. Make sure your speakers/headphones are listed FIRST (drag to reorder)"
echo "  6. Rename it to 'Meeting Scribe Output' (right-click the device name)"
echo ""
echo "When you want to transcribe a meeting:"
echo "  - Set your system output to 'Meeting Scribe Output'"
echo "  - Meeting Scribe will capture from 'BlackHole 2ch'"
echo ""
echo "=== Setup Complete ==="
