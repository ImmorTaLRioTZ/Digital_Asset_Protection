"""
Stage 3 — SSCD Heavy Artillery
================================
Loads a TorchScript SSCD vision model, generates L2-normalised 512-D
embeddings for every keyframe, then runs a vectorised sliding-window
cosine-similarity search to find the suspect clip inside the reference.

Catches: visually altered videos that defeated pHash AND had their audio
         replaced — heavy crops, text overlays, picture-in-picture,
         colour grading, and re-encoded derivatives.

Does NOT run unless Stages 1 and 2 both pass (GPU stays idle otherwise).

Dependencies: torch, torchvision, opencv-python-headless, Pillow
Optional    : tqdm  (progress bars)
External    : sscd_downloader  (auto-fetches the .torchscript weights)
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from pipeline.sscd_downloader import ensure_sscd_model_exists

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoFingerprint:
    """SSCD embedding fingerprint for one official reference video."""
    asset_id: str
    vectors: torch.Tensor          # (N, D) float32 on CPU — L2-normalised
    fps_interval: float            # seconds between sampled frames
    frame_count: int               # number of embedding vectors stored
    duration_seconds: float


@dataclass
class SSCDResult:
    matched: bool
    asset_id: Optional[str] = None
    best_score: float = 0.0        # mean cosine similarity of best window (0–1)
    best_timestamp_seconds: int = -1  # start second of the matching window
    reason: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SSCD_INPUT_SIZE   = 288           # model's expected square input resolution
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.90          # cosine similarity ≥ this → confirmed match
FPS_INTERVAL      = 1.0           # sample one frame per second by default
VRAM_OVERHEAD_MB  = 1_800         # ~1.5 GB reserved for model weights + driver


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s  (%d MB VRAM)", p.name, p.total_memory // 1024 ** 2)
    else:
        logger.warning("No CUDA GPU — running SSCD on CPU (will be slow).")
    return device


def _auto_batch_size(device: str, use_fp16: bool) -> int:
    """Derives the largest safe batch size from live VRAM headroom."""
    if device != "cuda":
        return 8
    try:
        total_mb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        free_mb   = max(total_mb - VRAM_OVERHEAD_MB, 256)
        bytes_per = SSCD_INPUT_SIZE * SSCD_INPUT_SIZE * 3 * (2 if use_fp16 else 4)
        batch     = max(4, int((free_mb * 1024 ** 2) / (bytes_per * 10)))
        batch     = min(batch, 128)
        logger.info("Auto batch size: %d  (%.0f MB free VRAM, fp%s)",
                    batch, free_mb, "16" if use_fp16 else "32")
        return batch
    except Exception:
        logger.warning("VRAM query failed — defaulting to batch_size=16.")
        return 16


def _load_model(
    model_path: Path,
    device: str,
    use_fp16: bool,
) -> torch.jit.ScriptModule:
    model_path = Path(model_path)
    logger.info("Loading SSCD TorchScript model from %s …", model_path.name)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    if use_fp16 and device == "cuda":
        model = model.half()
        logger.info("Model cast to FP16.")
    return model


_preprocess = transforms.Compose([
    transforms.Resize([SSCD_INPUT_SIZE, SSCD_INPUT_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _extract_keyframes(
    video_path: str | Path,
    fps_interval: float = FPS_INTERVAL,
) -> list[Image.Image]:
    """
    Sample one PIL Image every *fps_interval* seconds.
    Corrupted frames are skipped rather than crashing the pipeline.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        logger.warning("FPS metadata invalid (%.2f) — assuming 25 fps.", video_fps)
        video_fps = 25.0

    frame_interval = max(1, round(video_fps * fps_interval))
    total          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Source: %.2f fps | sample every %d frames | ~%d keyframes expected",
                video_fps, frame_interval, max(1, total // frame_interval))

    frames, current, skipped = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current % frame_interval == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            except Exception as exc:
                logger.warning("Skipping corrupt frame %d: %s", current, exc)
                skipped += 1
        current += 1

    cap.release()
    if skipped:
        logger.warning("%d corrupt frame(s) skipped.", skipped)
    logger.debug("Extracted %d keyframes from %s", len(frames), video_path)
    return frames


@torch.no_grad()
def _encode_frames(
    frames:     list[Image.Image],
    model:      torch.jit.ScriptModule,
    device:     str,
    batch_size: int,
    use_fp16:   bool,
) -> torch.Tensor:
    """
    Encode PIL frames → L2-normalised SSCD embeddings (float32, CPU).
    Moves results off the GPU after every batch to minimise peak VRAM usage.
    """
    dtype     = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
    batches   = range(0, len(frames), batch_size)
    iterator  = tqdm(batches, desc="  encoding", unit="batch") if _TQDM else batches
    collected = []

    for i in iterator:
        batch   = frames[i : i + batch_size]
        tensors = torch.stack([_preprocess(f) for f in batch]).to(device=device, dtype=dtype)
        vecs    = model(tensors)
        vecs    = F.normalize(vecs.float(), p=2, dim=1)   # fp32 for stable norm
        collected.append(vecs.cpu())
        del tensors, vecs
        if device == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(collected, dim=0)   # (N, D) float32 on CPU


def _sliding_window_search(
    ref_vectors:     torch.Tensor,   # (N, D) — official reference
    suspect_vectors: torch.Tensor,   # (M, D) — clip under investigation
    threshold:       float,
) -> Tuple[float, int]:
    """
    Vectorised sliding-window cosine similarity via grouped conv1d.

    Since all vectors are L2-normalised, dot(a, b) == cosine_sim(a, b).
    We rewrite the per-window mean dot product as a 1-D grouped convolution:

        signal = ref.T        → (1, D, N)
        kernel = suspect.T    → (D, 1, M)   (D independent filters)

    F.conv1d(groups=D) produces (1, D, N−M+1) in one fused CUDA/C++ call.
    Summing over D and dividing by M gives the mean cosine similarity for
    every window — no Python loop over windows.
    """
    N, D = ref_vectors.shape
    M    = suspect_vectors.shape[0]

    if M == 0 or N == 0:
        logger.error("Empty vector sequence — aborting search.")
        return 0.0, -1
    if M > N:
        logger.error(
            "Suspect clip (%d frames) is longer than reference (%d frames). "
            "Match is impossible.", M, N
        )
        return 0.0, -1

    logger.info("Sliding-window: %d windows × %d-frame clip inside %d-frame reference …",
                N - M + 1, M, N)

    signal       = ref_vectors.T.unsqueeze(0).float()        # (1, D, N)
    kernel       = suspect_vectors.T.unsqueeze(1).float()    # (D, 1, M)
    conv_out     = F.conv1d(signal, kernel, groups=D)        # (1, D, N-M+1)
    window_scores = conv_out.squeeze(0).sum(dim=0) / M       # (N-M+1,)

    best_score_t, best_idx_t = window_scores.max(dim=0)
    return best_score_t.item(), best_idx_t.item()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_fingerprint(
    video_path: str | Path,
    asset_id:   str,
    model_dir:  str | Path,
    fps_interval: float = FPS_INTERVAL,
    use_fp16:   bool    = True,
) -> VideoFingerprint:
    """
    Generate and return an SSCD embedding fingerprint for an *official*
    reference video. Store the result; pass it to `check_video` later.

    The model is loaded, used, and unloaded in a single call so VRAM
    is not held between fingerprint registrations.
    """
    path   = Path(video_path)
    device = _get_device()
    mpath  = ensure_sscd_model_exists(Path(model_dir))
    model  = _load_model(mpath, device, use_fp16)
    batch  = _auto_batch_size(device, use_fp16)

    frames = _extract_keyframes(path, fps_interval)
    if not frames:
        raise RuntimeError(f"No usable frames extracted from: {path}")

    vectors = _encode_frames(frames, model, device, batch, use_fp16)

    # Free model immediately — embedding generation is done
    del model, frames
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Estimate duration from frame count and interval
    cap  = cv2.VideoCapture(str(path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fc   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur  = (fc / fps) if fps else 0.0
    cap.release()

    fp = VideoFingerprint(
        asset_id=asset_id,
        vectors=vectors,
        fps_interval=fps_interval,
        frame_count=vectors.shape[0],
        duration_seconds=dur,
    )
    logger.info(
        "[Stage 3] Built SSCD fingerprint for '%s': %d vectors, dim=%d",
        asset_id, fp.frame_count, vectors.shape[1],
    )
    return fp


def check_video(
    video_path: str | Path,
    database:   list[VideoFingerprint],
    model_dir:  str | Path,
    threshold:  float = DEFAULT_THRESHOLD,
    fps_interval: float = FPS_INTERVAL,
    use_fp16:   bool    = True,
) -> SSCDResult:
    """
    Compare *video_path* against every SSCD fingerprint in *database*.

    The model is loaded once, the suspect video is encoded, then the model
    is unloaded before running the (CPU-only) sliding-window search. This
    ensures VRAM is freed regardless of how many assets are in the database.

    Returns an SSCDResult. If `matched=True`, the pipeline flags the video.
    If `matched=False`, the video passes all three stages and is marked clean.

    Algorithm
    ---------
    For each reference fingerprint we run a vectorised grouped conv1d
    sliding-window search (see `_sliding_window_search`). Because all
    vectors are L2-normalised, dot product == cosine similarity. The window
    with the highest mean similarity is used as the match score. A score
    above *threshold* confirms piracy even under heavy visual manipulation.
    """
    path = Path(video_path)
    logger.info("[Stage 3] Checking %s against %d SSCD fingerprints", path.name, len(database))

    # ── Encode the suspect video ──────────────────────────────────────────────
    device = _get_device()
    try:
        mpath  = ensure_sscd_model_exists(Path(model_dir))
        model  = _load_model(mpath, device, use_fp16)
        batch  = _auto_batch_size(device, use_fp16)
        frames = _extract_keyframes(path, fps_interval)
    except Exception as exc:
        return SSCDResult(matched=False, reason=f"Setup failed: {exc}")

    if not frames:
        return SSCDResult(matched=False, reason="No usable frames extracted from suspect video")

    try:
        suspect_vectors = _encode_frames(frames, model, device, batch, use_fp16)
    except Exception as exc:
        return SSCDResult(matched=False, reason=f"Encoding failed: {exc}")
    finally:
        # Always unload model before the search phase
        del model, frames
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info("[Stage 3] Model unloaded. Running search on CPU …")

    # ── Sliding-window search against every reference fingerprint ─────────────
    best_result = SSCDResult(matched=False, reason="No match found in SSCD database")

    for fp in database:
        score, best_idx = _sliding_window_search(fp.vectors, suspect_vectors, threshold)

        logger.debug(
            "  vs '%s': best_score=%.4f at window=%d (threshold=%.2f)",
            fp.asset_id, score, best_idx, threshold,
        )

        if score >= threshold and score > best_result.best_score:
            best_result = SSCDResult(
                matched=True,
                asset_id=fp.asset_id,
                best_score=score,
                best_timestamp_seconds=best_idx,  # 1 frame/s → frame index == second
                reason=(
                    f"SSCD matched '{fp.asset_id}' with cosine similarity {score:.4f} "
                    f"(threshold={threshold}) at ~{best_idx}s into reference"
                ),
            )

    if best_result.matched:
        logger.info("[Stage 3] ✓ MATCH — %s", best_result.reason)
    else:
        logger.info("[Stage 3] ✗ No match — video passes all stages (CLEAN)")

    return best_result