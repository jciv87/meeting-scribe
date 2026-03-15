"""Main controller and entry point for Meeting Scribe."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from meeting_scribe.audio.capture import AudioCapture
from meeting_scribe.config import load_config

logger = logging.getLogger(__name__)
from meeting_scribe.detection.monitor import MeetingMonitor
from meeting_scribe.diarization.embeddings import SpeakerEncoder
from meeting_scribe.diarization.matcher import SpeakerMatcher
from meeting_scribe.diarization.profiles import ProfileStore
from meeting_scribe.hotkey.listener import HotkeyListener
from meeting_scribe.output.markdown import MarkdownTranscript
from meeting_scribe.output.writer import TranscriptWriter
from meeting_scribe.transcription.engine import TranscriptionEngine
from meeting_scribe.transcription.streaming import TranscriptionWorker
from meeting_scribe.ui.menubar import MeetingScribeMenuBar

if TYPE_CHECKING:
    from meeting_scribe.config import Config


class MeetingScribeController:
    """Orchestrates audio capture, transcription, diarization, and output."""

    def __init__(self) -> None:
        self._config: Config = load_config()

        # Shared queues
        self.chunk_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()

        # Transcription
        self._engine = TranscriptionEngine(
            model_size=self._config.transcription.model_size,
            compute_type=self._config.transcription.compute_type,
            cpu_threads=self._config.transcription.cpu_threads,
            language=self._config.transcription.language,
            beam_size=self._config.transcription.beam_size,
        )
        self._worker = TranscriptionWorker(
            engine=self._engine,
            chunk_queue=self.chunk_queue,
            result_queue=self.result_queue,
            sample_rate=self._config.audio.sample_rate,
        )

        # Audio capture
        self._capture = AudioCapture(
            device_name=self._config.audio.device,
            sample_rate=self._config.audio.sample_rate,
            channels=self._config.audio.channels,
            chunk_duration=self._config.transcription.chunk_duration_seconds,
            overlap=self._config.transcription.overlap_seconds,
            chunk_queue=self.chunk_queue,
        )

        # Diarization
        self._encoder = SpeakerEncoder()
        self._profile_store = ProfileStore()
        self._matcher = SpeakerMatcher(
            profile_store=self._profile_store,
            encoder=self._encoder,
            threshold=self._config.diarization.similarity_threshold,
        )

        # Meeting detection
        self._monitor = MeetingMonitor(
            poll_interval=self._config.detection.poll_interval_seconds,
            start_debounce=self._config.detection.start_debounce_seconds,
            end_debounce=self._config.detection.end_debounce_seconds,
            on_meeting_start=self._on_meeting_detected,
            on_meeting_end=self._on_meeting_ended,
        )

        # Hotkey
        self._hotkey_listener = HotkeyListener(
            on_toggle=self._toggle,
            combination=self._config.hotkey.combination,
        )

        # Transcript state
        self._transcript: MarkdownTranscript | None = None
        self._writer: TranscriptWriter | None = None
        self._recording_start: float | None = None

        self._recording_lock = threading.Lock()
        self._is_recording = False

        # Silence detection
        self._silence_cycles = 0

        # Background result processor
        self._processor_stop = threading.Event()
        self._processor_thread = threading.Thread(
            target=self._result_loop,
            daemon=True,
            name="ResultProcessor",
        )
        self._processor_thread.start()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def detected_meeting(self) -> str | None:
        return self._monitor.current_meeting

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start(self, source: str | None = None) -> None:
        """Start audio capture, transcription worker, and a fresh transcript."""
        with self._recording_lock:
            if self._is_recording:
                return

            meeting_source = source or self._monitor.current_meeting or "Manual"

            self._transcript = MarkdownTranscript(
                source=meeting_source,
                start_time=datetime.now(),
            )
            self._writer = TranscriptWriter(
                output_dir=self._config.output.transcript_path,
                transcript=self._transcript,
            )
            self._recording_start = time.monotonic()
            self._is_recording = True

        self._worker.start()
        self._capture.start()

    def stop(self) -> None:
        """Stop capture, drain remaining results, save final transcript."""
        with self._recording_lock:
            if not self._is_recording:
                return
            self._is_recording = False

        self._capture.stop()
        self._worker.stop()

        # Drain any remaining results
        self._process_results()

        if self._transcript is not None:
            self._transcript.set_end_time(datetime.now())
        if self._writer is not None:
            self._writer.save_final()

            # Kick off summarization in the background
            if self._config.summarization.enabled:
                output_path = self._writer.get_output_path()
                threading.Thread(
                    target=self._summarize_transcript,
                    args=(output_path,),
                    daemon=True,
                    name="Summarizer",
                ).start()

    def _toggle(self) -> None:
        """Hotkey callback — flip recording state."""
        if self._is_recording:
            self.stop()
        else:
            self.start()

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_transcript(self, transcript_path: Path) -> None:
        """Generate a meeting summary in the background."""
        try:
            from meeting_scribe.summarization.engine import SummarizationEngine

            engine = SummarizationEngine(
                model=self._config.summarization.model,
                host=self._config.summarization.ollama_host,
                timeout=self._config.summarization.timeout_seconds,
            )
            summary_path = engine.summarize_file(transcript_path)

            try:
                import rumps
                rumps.notification(
                    title="Meeting Scribe",
                    subtitle="Summary ready",
                    message=str(summary_path.name),
                )
            except Exception:
                pass

            logger.info("Summarization complete: %s", summary_path)
        except Exception:
            logger.exception("Summarization failed for %s", transcript_path)
            try:
                import rumps
                rumps.notification(
                    title="Meeting Scribe",
                    subtitle="Summary failed",
                    message="Could not generate summary. Is Ollama running?",
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Meeting detection callbacks
    # ------------------------------------------------------------------

    def _on_meeting_detected(self, source: str) -> None:
        if self._config.ui.notify_on_detection:
            try:
                import rumps
                rumps.notification(
                    title="Meeting Scribe",
                    subtitle="Meeting detected",
                    message="Google Meet detected. Press Fn+F12 to start transcribing.",
                )
            except Exception:
                pass

        if self._config.ui.auto_start_recording and not self._is_recording:
            self.start(source=source)

    def _on_meeting_ended(self, source: str) -> None:
        saved_path: str | None = None
        if self._is_recording:
            if self._writer is not None:
                try:
                    saved_path = str(self._writer.output_path)
                except Exception:
                    pass
            self.stop()

        try:
            import rumps
            message = (
                f"Meeting ended. Transcript saved to {saved_path}"
                if saved_path
                else "Meeting ended."
            )
            rumps.notification(
                title="Meeting Scribe",
                subtitle="Meeting ended",
                message=message,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Result processing (background thread)
    # ------------------------------------------------------------------

    def _result_loop(self) -> None:
        while not self._processor_stop.is_set():
            self._process_results()
            self._processor_stop.wait(timeout=2)

    def _check_silence(self, had_speech: bool) -> None:
        """Track consecutive empty cycles and notify if silence threshold is exceeded."""
        if had_speech:
            self._silence_cycles = 0
            return

        self._silence_cycles += 1

        # Each _result_loop cycle is ~2 seconds; compute threshold in cycles
        timeout_seconds = self._config.detection.silence_timeout_seconds
        cycles_threshold = max(1, timeout_seconds // 2)

        if self._silence_cycles == cycles_threshold:
            try:
                import rumps
                rumps.notification(
                    title="Meeting Scribe",
                    subtitle="Silence detected",
                    message=f"No speech detected for {timeout_seconds} seconds. Meeting still active?",
                )
            except Exception:
                pass

    def _process_results(self) -> None:
        """Drain result_queue, run diarization, add entries to transcript."""
        if self._transcript is None:
            # Flush the queue even if there is no active transcript
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            return

        recording_start = self._recording_start or time.monotonic()
        had_speech = False

        while not self.result_queue.empty():
            try:
                segments = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if not isinstance(segments, list):
                continue

            for segment in segments:
                had_speech = True
                speaker = "Unknown"
                if segment.audio_chunk is not None:
                    try:
                        speaker, _ = self._matcher.identify(segment.audio_chunk)
                    except Exception:
                        pass

                # segment.start is relative to the start of its audio chunk;
                # we want wall-clock offset from meeting start
                elapsed = segment.start
                self._transcript.add_entry(
                    timestamp=elapsed,
                    speaker=speaker,
                    text=segment.text.strip(),
                )

            # Incremental save
            if self._writer is not None:
                try:
                    self._writer.save_partial()
                except Exception:
                    pass

        if self._is_recording:
            self._check_silence(had_speech)


def main() -> None:
    """Create all components and run the menu bar application."""
    controller = MeetingScribeController()
    menubar = MeetingScribeMenuBar(controller=controller)

    controller._monitor.start()
    controller._hotkey_listener.start()

    menubar.run()  # Blocks until quit


if __name__ == "__main__":
    main()
