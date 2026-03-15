# Meeting Scribe

macOS menu bar app that transcribes meetings — Google Meet, Zoom, Slack Huddles — with speaker diarization and learned voice profiles. Audio is captured via BlackHole's virtual audio device, so both remote and local speech are captured without a microphone. Transcription runs locally using faster-whisper (large-v3-turbo). Speaker identification uses SpeechBrain's ECAPA-TDNN model; after you label speakers from the first meeting, future meetings auto-label recognized voices. Transcripts save as Markdown to `~/meeting-scribe/transcripts/`.

---

## Prerequisites

- macOS (Apple Silicon)
- Python 3.12+
- Homebrew

---

## Installation

### 1. Clone or download the project

```bash
git clone <repo-url> ~/meeting-scribe
cd ~/meeting-scribe
```

### 2. Install BlackHole

BlackHole creates the virtual audio device Meeting Scribe captures from.

```bash
brew install blackhole-2ch
```

**Reboot after installation.** BlackHole requires a system restart before the device appears.

### 3. Create the Multi-Output Device

This routes system audio to both your speakers and BlackHole simultaneously.

1. Open **Audio MIDI Setup** (Spotlight: `Cmd+Space`, type "Audio MIDI Setup")
2. Click `+` at the bottom left and select **Create Multi-Output Device**
3. Check both:
   - Your speakers or headphones (e.g., "MacBook Pro Speakers", "AirPods")
   - **BlackHole 2ch**
4. Drag your speakers to the top of the list
5. Right-click the new device and rename it **Meeting Scribe Output**

> Set your system audio output to "Meeting Scribe Output" before each meeting so Meeting Scribe can capture the audio.

### 4. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you are running a local build of faster-whisper:

```bash
pip install -e ~/faster-whisper
```

---

## macOS Permissions

On first run, macOS will prompt for:

- **Accessibility** — required for the Fn+F12 global hotkey
- **Microphone** — required for audio capture from BlackHole
- **Automation** — required to detect active meetings via AppleScript

Grant all three when prompted. If a permission is denied, open System Settings > Privacy & Security and enable it manually.

---

## Usage

```bash
source .venv/bin/activate
python -m meeting_scribe.app
```

A menu bar icon appears. Meeting Scribe polls for active Google Meet, Zoom, and Slack Huddle sessions every 5 seconds. When a meeting is detected, a notification appears.

**Before the meeting starts:** set your system audio output to "Meeting Scribe Output".

| Action | Result |
|--------|--------|
| `Fn+F12` | Start or stop transcription |
| Meeting detected + `auto_start_recording: true` | Recording starts automatically |
| Meeting ends | Recording stops; transcript saved |

Transcripts are saved to `~/meeting-scribe/transcripts/` as Markdown files, with incremental saves every 60 seconds during recording. A silence notification fires if no speech is detected for 60 seconds while recording is active.

---

## Speaker Profiles

**First meeting:** speakers are labeled "Unknown", "Speaker A", "Speaker B", etc.

**After the meeting:** use the menu bar to assign names to each speaker label.

**Future meetings:** recognized voices are auto-labeled. Voice embeddings are stored in `profiles/voices.db`.

---

## Configuration

Edit `config.yaml` to change behavior. Key settings:

```yaml
transcription:
  model_size: "large-v3-turbo"   # Recommended for accented speech.
                                  # Smaller options: medium, small, base.

diarization:
  similarity_threshold: 0.75     # Speaker match confidence. Lower = more
                                  # liberal matching. Raise if speakers are
                                  # being confused; lower if known speakers
                                  # aren't being recognized.

ui:
  auto_start_recording: false    # false = press Fn+F12 to start (opt-in).
                                  # true = recording starts automatically
                                  # when a meeting is detected.
```

---

## Auto-Start at Login

To launch Meeting Scribe automatically when you log in:

```bash
cp com.meetingscribe.agent.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.meetingscribe.agent.plist
```

Logs are written to `logs/stdout.log` and `logs/stderr.log`.

To stop auto-start:

```bash
launchctl unload ~/Library/LaunchAgents/com.meetingscribe.agent.plist
```

---

## Project Structure

```
meeting-scribe/
  meeting_scribe/
    app.py              # Main controller and entry point
    config.py           # Config loading
    audio/              # BlackHole capture
    detection/          # Meeting monitor (Meet, Zoom, Huddles)
    diarization/        # Speaker embeddings, matching, profiles
    hotkey/             # Fn+F12 listener
    output/             # Markdown transcript format and writer
    transcription/      # faster-whisper engine and streaming worker
    ui/                 # Menu bar (rumps)
  profiles/
    voices.db           # Learned speaker embeddings
  transcripts/          # Saved Markdown transcripts
  logs/                 # stdout/stderr from LaunchAgent
  config.yaml           # User configuration
  setup_audio.sh        # One-time audio setup helper
  com.meetingscribe.agent.plist  # LaunchAgent for login auto-start
  pyproject.toml
```
