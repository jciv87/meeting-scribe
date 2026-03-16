# Meeting Scribe

Local macOS meeting transcription app. Audio via BlackHole, transcription via faster-whisper, speaker diarization via SpeechBrain ECAPA-TDNN, summarization via Ollama. No cloud services.

## Architecture

Controller-orchestrator pattern (`app.py`) with queue-based audio pipeline:
`AudioCapture → chunk_queue → TranscriptionWorker → result_queue → ResultProcessor (diarize + write)`

Post-meeting: optional Ollama summarization in background thread.

## Key Files

- `meeting_scribe/app.py` — Main controller, wires all subsystems
- `meeting_scribe/config.py` — Dataclass config with defaults, loads from `config.yaml`
- `config.yaml` — Runtime configuration (single source of truth)

## Commands

```bash
# Run the app
meeting-scribe

# Install (editable)
pip install -e .
```

## Conventions

- All config sections have dataclass defaults in `config.py` — missing YAML sections fall back gracefully
- Thread-safe: recording state protected by `_recording_lock`
- Audio pipeline uses `queue.Queue` for thread communication
- Transcripts saved as Markdown to `~/meeting-scribe/transcripts/`
- Speaker profiles stored in `profiles/voices.db` (SQLite)

## Boundaries

- **Local-only**: No cloud services, no network calls except Ollama on localhost
- **macOS-only**: Depends on BlackHole, AppleScript, rumps
- **Python >=3.12**: Uses modern type syntax (`X | Y`, `from __future__ import annotations`)
