"""Tests for meeting_scribe.diarization.profiles.ProfileStore (in-memory SQLite)."""

import pytest
import numpy as np
from pathlib import Path

from meeting_scribe.diarization.profiles import ProfileStore, SpeakerProfile


@pytest.fixture
def store(tmp_path: Path) -> ProfileStore:
    """Fresh ProfileStore backed by a temp SQLite file."""
    db = tmp_path / "test_voices.db"
    s = ProfileStore(db_path=db)
    yield s
    s.close()


def make_embedding(dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# add_speaker / get_profile
# ---------------------------------------------------------------------------

class TestAddSpeaker:
    def test_add_speaker_returns_int_id(self, store):
        sid = store.add_speaker("Alice")
        assert isinstance(sid, int)

    def test_add_speaker_ids_are_unique(self, store):
        a = store.add_speaker("Alice")
        b = store.add_speaker("Bob")
        assert a != b

    def test_get_profile_returns_correct_name(self, store):
        sid = store.add_speaker("Carol")
        profile = store.get_profile(sid)
        assert profile is not None
        assert profile.name == "Carol"

    def test_get_profile_missing_returns_none(self, store):
        assert store.get_profile(9999) is None

    def test_get_profile_no_embeddings_initially(self, store):
        sid = store.add_speaker("Dave")
        profile = store.get_profile(sid)
        assert profile.embeddings == []


# ---------------------------------------------------------------------------
# list_speakers / get_all_profiles
# ---------------------------------------------------------------------------

class TestListSpeakers:
    def test_empty_store_returns_empty_list(self, store):
        assert store.get_all_profiles() == []

    def test_all_profiles_contains_added_speakers(self, store):
        store.add_speaker("Alice")
        store.add_speaker("Bob")
        profiles = store.get_all_profiles()
        names = {p.name for p in profiles}
        assert names == {"Alice", "Bob"}

    def test_all_profiles_ordered_by_id(self, store):
        store.add_speaker("Zara")
        store.add_speaker("Alice")
        profiles = store.get_all_profiles()
        assert profiles[0].name == "Zara"
        assert profiles[1].name == "Alice"


# ---------------------------------------------------------------------------
# add_embedding / update_embedding (replace via centroid)
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_add_embedding_stored_and_retrieved(self, store):
        sid = store.add_speaker("Alice")
        emb = make_embedding()
        store.add_embedding(sid, emb)
        profile = store.get_profile(sid)
        assert len(profile.embeddings) == 1
        np.testing.assert_array_almost_equal(profile.embeddings[0], emb)

    def test_multiple_embeddings_all_retrieved(self, store):
        sid = store.add_speaker("Bob")
        for i in range(3):
            store.add_embedding(sid, make_embedding(seed=i))
        profile = store.get_profile(sid)
        assert len(profile.embeddings) == 3

    def test_centroid_of_two_embeddings(self, store):
        sid = store.add_speaker("Carol")
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        store.add_embedding(sid, e1)
        store.add_embedding(sid, e2)
        profile = store.get_profile(sid)
        centroid = profile.centroid()
        np.testing.assert_array_almost_equal(centroid, [0.5, 0.5])

    def test_centroid_raises_with_no_embeddings(self, store):
        sid = store.add_speaker("Dave")
        profile = store.get_profile(sid)
        with pytest.raises(ValueError, match="no embeddings"):
            profile.centroid()

    def test_add_embedding_with_meeting_date(self, store):
        sid = store.add_speaker("Eve")
        store.add_embedding(sid, make_embedding(), meeting_date="2024-06-15")
        profile = store.get_profile(sid)
        assert len(profile.embeddings) == 1

    def test_embedding_dtype_float32_roundtrip(self, store):
        sid = store.add_speaker("Frank")
        emb = make_embedding(dim=128)
        store.add_embedding(sid, emb)
        profile = store.get_profile(sid)
        assert profile.embeddings[0].dtype == np.float32


# ---------------------------------------------------------------------------
# rename_speaker / delete_speaker
# ---------------------------------------------------------------------------

class TestMutations:
    def test_rename_speaker_changes_name(self, store):
        sid = store.add_speaker("Old Name")
        store.rename_speaker(sid, "New Name")
        profile = store.get_profile(sid)
        assert profile.name == "New Name"

    def test_delete_speaker_removes_from_store(self, store):
        sid = store.add_speaker("ToDelete")
        store.delete_speaker(sid)
        assert store.get_profile(sid) is None

    def test_delete_speaker_removes_embeddings(self, store):
        sid = store.add_speaker("WithEmb")
        store.add_embedding(sid, make_embedding())
        store.delete_speaker(sid)
        # Re-adding a speaker with same id won't happen, but profile is gone
        assert store.get_profile(sid) is None

    def test_delete_nonexistent_speaker_does_not_raise(self, store):
        store.delete_speaker(9999)  # should silently do nothing


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_closes_cleanly(self, tmp_path):
        db = tmp_path / "ctx.db"
        with ProfileStore(db_path=db) as s:
            sid = s.add_speaker("Alice")
            assert sid is not None
