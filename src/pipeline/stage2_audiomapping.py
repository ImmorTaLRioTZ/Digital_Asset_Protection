"""
Stage 2 — Audio Fingerprint Detective
======================================
Extracts audio from a video, builds a Shazam-style landmark fingerprint
(constellation map over a log-frequency spectrogram), and compares it
against a reference fingerprint using an offset-histogram vote.

Catches: visually-flipped/heavily-cropped videos whose audio is intact.
         Robust to re-encoding, mild EQ changes, and slight pitch shifts.
Does NOT catch: muted videos or completely replaced audio → Stage 3.

Dependencies: librosa, numpy, scipy, soundfile (transitive via librosa)
Audio extraction: ffmpeg must be installed on the host (usually already present).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE         = 22050   # Hz — enough for fingerprinting, keeps CPU load low
HOP_LENGTH          = 512     # STFT hop in samples
N_FFT               = 2048    # STFT window size
N_MELS              = 128     # mel bands
TOP_PEAKS_PER_FRAME = 3       # strongest peaks per time slice kept as landmarks
FAN_OUT             = 5       # each peak paired with next FAN_OUT peaks → hash
MIN_OFFSET          = 1       # minimum time-frame gap between paired peaks
MAX_OFFSET          = 50      # maximum time-frame gap between paired peaks
VOTE_THRESHOLD      = 5       # minimum landmark votes to confirm a match


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioFingerprint:
    """Serialisable audio fingerprint for one reference asset."""
    asset_id: str
    hashes: dict[str, list[int]]   # hash_hex → list of time offsets (frames)
    duration_seconds: float


@dataclass
class AudioResult:
    matched: bool
    asset_id: Optional[str] = None
    vote_count: int = 0
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Audio extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio(video_path: str | Path) -> np.ndarray:
    """
    Use ffmpeg to strip audio from *video_path* and return a mono float32
    numpy array at SAMPLE_RATE.

    Falls back to librosa's native loader if ffmpeg is unavailable (slower).
    """
    path = Path(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ac", "1",                      # mono
        "-ar", str(SAMPLE_RATE),         # resample
        "-vn",                           # drop video stream
        "-f", "wav", tmp_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        logger.debug("ffmpeg extracted %.1f s of audio", len(y) / SAMPLE_RATE)
        return y
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg not available — fall back to librosa (supports many containers)
        logger.warning("ffmpeg unavailable; using librosa audio backend (slower)")
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return y
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark / constellation-map fingerprinting (Shazam-style)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Return a log-power mel spectrogram (shape: n_mels × n_frames)."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return librosa.power_to_db(S, ref=np.max)


def _find_peaks(spec: np.ndarray) -> list[tuple[int, int]]:
    """
    Find the locally loudest (peak) bins across the spectrogram.
    Returns a list of (time_frame, mel_bin) tuples sorted by time.
    """
    from scipy.ndimage import maximum_filter

    # A point is a peak if it equals the local max in a neighbourhood
    neighbourhood = maximum_filter(spec, size=(5, 5))
    peak_mask     = (spec == neighbourhood)

    # Keep only the loudest TOP_PEAKS_PER_FRAME per time slice
    n_frames = spec.shape[1]
    peaks: list[tuple[int, int]] = []
    for t in range(n_frames):
        col_peaks = np.where(peak_mask[:, t])[0]
        if len(col_peaks) == 0:
            continue
        # Sort by amplitude descending, keep top N
        top = sorted(col_peaks, key=lambda b: spec[b, t], reverse=True)
        for b in top[:TOP_PEAKS_PER_FRAME]:
            peaks.append((t, int(b)))

    return peaks


def _hash_pairs(peaks: list[tuple[int, int]]) -> dict[str, list[int]]:
    """
    Create a hash table: hash_hex → [anchor_time, ...].

    Each anchor peak is paired with the next FAN_OUT peaks within
    [MIN_OFFSET, MAX_OFFSET] time frames. The triplet
    (anchor_freq, target_freq, time_delta) is hashed to a 32-bit hex string.
    This makes the fingerprint shift-invariant (doesn't depend on absolute time).
    """
    hash_table: dict[str, list[int]] = defaultdict(list)

    for i, (t1, f1) in enumerate(peaks):
        for j in range(i + 1, min(i + 1 + FAN_OUT, len(peaks))):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < MIN_OFFSET or dt > MAX_OFFSET:
                continue
            # Pack into a single integer and hex-encode
            raw   = f"{f1}|{f2}|{dt}"
            h     = hashlib.md5(raw.encode()).hexdigest()[:8]
            hash_table[h].append(t1)

    return dict(hash_table)


def _build_fingerprint_from_audio(y: np.ndarray, asset_id: str) -> AudioFingerprint:
    spec   = _compute_mel_spectrogram(y)
    peaks  = _find_peaks(spec)
    hashes = _hash_pairs(peaks)
    dur    = len(y) / SAMPLE_RATE
    logger.debug("Fingerprinted '%s': %d peaks → %d hashes", asset_id, len(peaks), len(hashes))
    return AudioFingerprint(asset_id=asset_id, hashes=hashes, duration_seconds=dur)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_fingerprint(video_path: str | Path, asset_id: str) -> AudioFingerprint:
    """
    Build and store an audio fingerprint for an *official* reference video.
    Call this once per asset; persist the result in your database.
    """
    y = _extract_audio(video_path)
    fp = _build_fingerprint_from_audio(y, asset_id)
    logger.info(
        "[Stage 2] Built audio fingerprint for '%s': %.1f s, %d hashes",
        asset_id, fp.duration_seconds, len(fp.hashes),
    )
    return fp


def check_video(
    video_path: str | Path,
    database: list[AudioFingerprint],
    vote_threshold: int = VOTE_THRESHOLD,
) -> AudioResult:
    """
    Compare the audio of *video_path* against every fingerprint in *database*.

    Uses an offset-histogram vote: matching hashes from query and reference
    that share the same *time delta* between them cast a vote. A high vote
    count means the audio is structurally identical regardless of re-encoding
    or mild EQ changes.

    Returns AudioResult. If `matched=True`, stop the pipeline and flag.
    If `matched=False`, escalate to Stage 3.
    """
    path = Path(video_path)
    logger.info("[Stage 2] Checking audio of %s", path.name)

    try:
        y = _extract_audio(path)
    except Exception as exc:
        return AudioResult(matched=False, reason=f"Audio extraction failed: {exc}")

    if y is None or len(y) < SAMPLE_RATE:
        return AudioResult(matched=False, reason="Audio too short or silent — escalating to Stage 3")

    # Build query fingerprint
    query_fp = _build_fingerprint_from_audio(y, asset_id="<query>")

    if not query_fp.hashes:
        return AudioResult(matched=False, reason="No audio landmarks found — video may be silent")

    best_result = AudioResult(matched=False, reason="No match found in audio fingerprint database")

    for ref_fp in database:
        # Offset histogram: for each shared hash key, compute time-delta between
        # query anchor time and reference anchor time. A strong peak in this
        # histogram means the audio is time-aligned and therefore the same track.
        offset_votes: dict[int, int] = defaultdict(int)

        for h, query_times in query_fp.hashes.items():
            if h not in ref_fp.hashes:
                continue
            for qt in query_times:
                for rt in ref_fp.hashes[h]:
                    offset_votes[rt - qt] += 1

        if not offset_votes:
            continue

        best_offset, best_votes = max(offset_votes.items(), key=lambda kv: kv[1])

        logger.debug(
            "  vs '%s': best_votes=%d at offset=%d frames",
            ref_fp.asset_id, best_votes, best_offset,
        )

        if best_votes >= vote_threshold and best_votes > best_result.vote_count:
            best_result = AudioResult(
                matched=True,
                asset_id=ref_fp.asset_id,
                vote_count=best_votes,
                reason=(
                    f"Audio matched '{ref_fp.asset_id}' with {best_votes} landmark votes "
                    f"at time-offset {best_offset} frames"
                ),
            )

    if best_result.matched:
        logger.info("[Stage 2] ✓ MATCH — %s", best_result.reason)
    else:
        logger.info("[Stage 2] ✗ No audio match — escalating to Stage 3")

    return best_result