"""
Stage 1 — pHash Bouncer
=======================
Extracts a sparse set of evenly-spaced keyframes from a video, hashes each
one with a 64-bit perceptual hash, and compares every hash against a database
of known-good fingerprints using Hamming Distance.

Catches: direct re-uploads, re-encodes, mild colour grading, slight crops,
         resolution changes, and compression artefacts.
Does NOT catch: horizontal flips, heavy crops, audio-only tricks → Stage 2.

Dependencies: opencv-python-headless, imagehash, Pillow, numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoFingerprint:
    """Serialisable pHash fingerprint for one video asset."""
    asset_id: str
    frame_hashes: list[str]          # hex strings, one per sampled frame
    frame_count: int
    fps: float
    duration_seconds: float


@dataclass
class PHashResult:
    matched: bool
    asset_id: Optional[str] = None
    best_distance: int = 999          # lower = more similar (0 = identical)
    matched_frame_ratio: float = 0.0  # fraction of frames that matched
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HAMMING_THRESHOLD     = 10    # bits different; ≤10 on a 64-bit hash = very similar
MATCH_FRAME_RATIO     = 0.40  # ≥40 % of frames must individually match
FRAMES_TO_SAMPLE      = 16    # how many evenly-spaced frames to extract
HASH_SIZE             = 8     # imagehash hash_size → 8×8 = 64-bit hash


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str | Path, n_frames: int = FRAMES_TO_SAMPLE) -> list[Image.Image]:
    """
    Extract *n_frames* evenly-spaced frames from a video file.
    Returns a list of PIL Images (RGB).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Could not determine frame count for: {video_path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames: list[Image.Image] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    logger.debug("Extracted %d frames from %s", len(frames), video_path)
    return frames


def _hash_frame(img: Image.Image) -> imagehash.ImageHash:
    """Return a 64-bit perceptual hash for one frame."""
    return imagehash.phash(img, hash_size=HASH_SIZE)


def _hamming(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2   # imagehash overloads subtraction to return Hamming distance


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_fingerprint(video_path: str | Path, asset_id: str) -> VideoFingerprint:
    """
    Build and return a pHash fingerprint for an *official* reference video.
    Store the result in your database; pass it to `check_video` later.
    """
    path = Path(video_path)
    cap  = cv2.VideoCapture(str(path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fc   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur  = (fc / fps) if fps else 0.0
    cap.release()

    frames = _extract_frames(path, FRAMES_TO_SAMPLE)
    hashes = [str(_hash_frame(f)) for f in frames]

    fp = VideoFingerprint(
        asset_id=asset_id,
        frame_hashes=hashes,
        frame_count=fc,
        fps=fps,
        duration_seconds=dur,
    )
    logger.info("Built pHash fingerprint for '%s': %d hashes", asset_id, len(hashes))
    return fp


def check_video(
    video_path: str | Path,
    database: list[VideoFingerprint],
    hamming_threshold: int = HAMMING_THRESHOLD,
    match_ratio_threshold: float = MATCH_FRAME_RATIO,
) -> PHashResult:
    """
    Compare *video_path* against every fingerprint in *database*.

    Returns a PHashResult. If `matched=True` the caller should stop the
    pipeline and flag the video. If `matched=False`, escalate to Stage 2.

    Algorithm
    ---------
    For each DB fingerprint we align the query frames 1-to-1 with the
    reference frames (by temporal position) and count how many pairs fall
    within *hamming_threshold*. A video is flagged when ≥ *match_ratio_threshold*
    of its frames match the reference.
    """
    path = Path(video_path)
    logger.info("[Stage 1] Checking %s against %d fingerprints", path.name, len(database))

    try:
        query_frames = _extract_frames(path, FRAMES_TO_SAMPLE)
    except (IOError, ValueError) as exc:
        return PHashResult(matched=False, reason=f"Frame extraction failed: {exc}")

    query_hashes = [_hash_frame(f) for f in query_frames]

    best_result = PHashResult(matched=False, reason="No match found in pHash database")

    for fp in database:
        ref_hashes = [imagehash.hex_to_hash(h) for h in fp.frame_hashes]

        # Pair up by index (both sides were sampled at equal temporal intervals)
        pairs = min(len(query_hashes), len(ref_hashes))
        if pairs == 0:
            continue

        distances   = [_hamming(query_hashes[i], ref_hashes[i]) for i in range(pairs)]
        matched_cnt = sum(1 for d in distances if d <= hamming_threshold)
        ratio       = matched_cnt / pairs
        best_dist   = min(distances)

        logger.debug(
            "  vs '%s': best_dist=%d, match_ratio=%.2f (%d/%d frames)",
            fp.asset_id, best_dist, ratio, matched_cnt, pairs,
        )

        if ratio >= match_ratio_threshold:
            result = PHashResult(
                matched=True,
                asset_id=fp.asset_id,
                best_distance=best_dist,
                matched_frame_ratio=ratio,
                reason=(
                    f"pHash matched {matched_cnt}/{pairs} frames "
                    f"(ratio={ratio:.2f}, best_hamming={best_dist})"
                ),
            )
            # Keep the strongest match
            if result.matched_frame_ratio > best_result.matched_frame_ratio:
                best_result = result

    if best_result.matched:
        logger.info("[Stage 1] ✓ MATCH — %s", best_result.reason)
    else:
        logger.info("[Stage 1] ✗ No match — escalating to Stage 2")

    return best_result