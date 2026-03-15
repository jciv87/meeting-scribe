import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from meeting_scribe.audio.buffer import AudioRingBuffer


class AudioCapture:
    """Capture audio from a named device and push numpy chunks onto a queue."""

    def __init__(
        self,
        device_name: str = "BlackHole 2ch",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: int = 30,
        overlap: int = 2,
        chunk_queue: Optional[queue.Queue] = None,
    ) -> None:
        if chunk_queue is None:
            raise ValueError("chunk_queue is required")

        self._device_name = device_name
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_duration = chunk_duration
        self._overlap = overlap
        self._chunk_queue: queue.Queue = chunk_queue

        self._buffer = AudioRingBuffer(
            capacity_seconds=max(chunk_duration * 4, 300),
            sample_rate=sample_rate,
        )
        self._running = threading.Event()
        self._stream: Optional[sd.InputStream] = None

    def _resolve_device(self) -> int:
        """Return the device index for *device_name*.

        Raises ``ValueError`` if no matching device is found.
        """
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if self._device_name.lower() in dev["name"].lower():
                return idx
        raise ValueError(
            f"Audio device '{self._device_name}' not found. "
            f"Available devices: {[d['name'] for d in devices]}"
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        if not self._running.is_set():
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        self._buffer.write(mono)
        if self._buffer.pending_seconds() >= self._chunk_duration:
            chunk = self._buffer.read_chunk(self._chunk_duration, self._overlap)
            if chunk.size > 0:
                try:
                    self._chunk_queue.put_nowait((chunk, self._chunk_duration))
                except queue.Full:
                    pass

    def start(self) -> None:
        """Open the input stream and begin capturing audio."""
        if self._running.is_set():
            return
        device_idx = self._resolve_device()
        self._running.set()
        self._stream = sd.InputStream(
            device=device_idx,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing and close the stream."""
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
