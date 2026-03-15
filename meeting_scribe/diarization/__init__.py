"""Speaker diarization package for meeting-scribe.

Heavy dependencies (torch, speechbrain) are imported lazily inside
SpeakerEncoder so that ProfileStore and MeetingClusterer remain usable
even when those optional packages are not installed.
"""

from .cluster import MeetingClusterer
from .profiles import ProfileStore, SpeakerProfile

__all__ = [
    "MeetingClusterer",
    "ProfileStore",
    "SpeakerProfile",
    # SpeakerEncoder and SpeakerMatcher require torch + speechbrain;
    # import them explicitly when needed:
    #   from meeting_scribe.diarization.embeddings import SpeakerEncoder
    #   from meeting_scribe.diarization.matcher import SpeakerMatcher
]
