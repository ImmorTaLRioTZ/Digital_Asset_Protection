"""
SSCD Video Piracy Detector  —  Rectified & Optimised
=====================================================
Key improvements over original:
  • Vectorised sliding-window via grouped conv1d  (no Python loop)
  • FP16 inference  → fits comfortably in 4 GB VRAM
  • Dynamic batch-size calculation from live VRAM headroom
  • GPU freed between encoding passes  (model unloaded before search)
  • Corrupted-frame guard  +  zero-FPS fallback
  • Progress bars via tqdm (optional)
  • Structured logging instead of raw print()
  • Explicit threshold forwarded through every call
  • Clean return type from main pipeline
"""

import gc
import logging
from pathlib import Path
from typing import Tuple

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

from sscd_downloader import ensure_sscd_model_exists

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sscd")

# ── Constants ─────────────────────────────────────────────────────────────────
SSCD_INPUT_SIZE   = 288
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
DEFAULT_THRESHOLD = 0.90
# We reserve this much VRAM for the OS / driver / model weights.
# Anything above it is available for image batches.
VRAM_OVERHEAD_MB  = 1_800   # ~1.5 GB for model + driver slack on a 4 GB card


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Hardware helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info("GPU  : %s  (%d MB VRAM)", p.name, p.total_memory // 1024 ** 2)
    else:
        log.warning("No CUDA GPU detected — running on CPU (will be slow).")
    return device


def _auto_batch_size(device: str, use_fp16: bool) -> int:
    """
    Derives the largest safe batch size from available VRAM.
    Falls back to a conservative value when on CPU or if the query fails.
    """
    if device != "cuda":
        return 8

    try:
        total_mb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        free_mb   = max(total_mb - VRAM_OVERHEAD_MB, 256)
        # Memory per 288×288 image tensor in bytes (3 channels)
        bytes_per = SSCD_INPUT_SIZE * SSCD_INPUT_SIZE * 3 * (2 if use_fp16 else 4)
        # Use a 10× overhead factor to account for activations + gradient buffers
        batch = max(4, int((free_mb * 1024 ** 2) / (bytes_per * 10)))
        batch = min(batch, 128)   # hard ceiling — prevent runaway allocation
        log.info("Auto batch size: %d  (%.0f MB free VRAM, fp%s)",
                 batch, free_mb, "16" if use_fp16 else "32")
        return batch
    except Exception:
        log.warning("VRAM query failed; defaulting to batch_size=16.")
        return 16


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_model(model_path: Path, device: str, use_fp16: bool) -> torch.jit.ScriptModule:
    log.info("Loading SSCD TorchScript model …")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # FP16 is safe for SSCD inference and halves VRAM consumption on GPU
    if use_fp16 and device == "cuda":
        model = model.half()
        log.info("Model cast to FP16.")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Frame extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keyframes(video_path: str, fps_interval: float = 1.0) -> list:
    """
    Returns a list of RGB PIL Images sampled at `fps_interval` seconds.
    Corrupted frames are skipped rather than crashing the pipeline.
    """
    log.info("Extracting keyframes from  %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        # Corrupt or missing FPS metadata — assume 25 fps (safe default)
        log.warning("FPS metadata is invalid (%.2f); falling back to 25 fps.", video_fps)
        video_fps = 25.0

    frame_interval = max(1, round(video_fps * fps_interval))
    total          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected       = max(1, total // frame_interval)
    log.info("Source FPS: %.2f | Sample every %d frames | ~%d keyframes expected",
             video_fps, frame_interval, expected)

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
                log.warning("Skipping corrupt frame %d: %s", current, exc)
                skipped += 1
        current += 1

    cap.release()
    if skipped:
        log.warning("%d corrupt frame(s) skipped.", skipped)
    log.info("Extracted %d keyframes.", len(frames))
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SSCD embedding
# ═══════════════════════════════════════════════════════════════════════════════

_preprocess = transforms.Compose([
    transforms.Resize([SSCD_INPUT_SIZE, SSCD_INPUT_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def get_sscd_vectors(
    frames:     list,
    model:      torch.jit.ScriptModule,
    device:     str,
    batch_size: int,
    use_fp16:   bool = True,
) -> torch.Tensor:
    """
    Encodes PIL frames → L2-normalised SSCD embeddings  (float32, on CPU).

    The output lives on CPU so that GPU memory is freed immediately after
    encoding, leaving headroom for the next video and for the search step.
    """
    dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

    batches   = range(0, len(frames), batch_size)
    iterator  = (tqdm(batches, desc="  encoding", unit="batch") if _TQDM else batches)
    collected = []

    for i in iterator:
        batch   = frames[i : i + batch_size]
        tensors = torch.stack([_preprocess(f) for f in batch]).to(device=device, dtype=dtype)

        vecs = model(tensors)
        # Cast back to fp32 before L2-norm to avoid precision loss during normalisation
        vecs = F.normalize(vecs.float(), p=2, dim=1)

        collected.append(vecs.cpu())   # move off GPU immediately

        del tensors, vecs
        if device == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(collected, dim=0)   # (N, D)  float32 on CPU


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Vectorised sliding-window  (the critical fix)
# ═══════════════════════════════════════════════════════════════════════════════

def find_clip_in_long_video(
    official_vectors: torch.Tensor,
    suspect_vectors:  torch.Tensor,
    threshold:        float = DEFAULT_THRESHOLD,
) -> Tuple[float, int]:
    """
    Finds the best-matching window of `suspect` inside `official`.

    ─── Algorithm ────────────────────────────────────────────────────────────
    Since all vectors are L2-normalised, cosine_sim(a, b) = dot(a, b).

    For window starting at offset i:
        score(i) = (1/M) · Σ_j  official[i+j] · suspect[j]    j=0..M-1

    Rewrite as a grouped 1-D cross-correlation across D=512 channels:

        signal  = official.T  → shape (1, D, N)     (D channels, length N)
        kernel  = suspect.T   → shape (D, 1, M)     (D filters,  width  M)

    F.conv1d with groups=D gives output (1, D, N-M+1).
    Summing over D and dividing by M yields the mean cosine similarity for
    every window in one single GPU/CPU call — no Python loop at all.

    Complexity:  O(N · M · D)  in C++/CUDA  vs  O(N · M · D) iterations of
    Python in the original — practically 100–1000× faster for long videos.
    ──────────────────────────────────────────────────────────────────────────
    """
    N, D = official_vectors.shape
    M    = suspect_vectors.shape[0]

    if M == 0 or N == 0:
        log.error("Empty vector sequence — aborting search.")
        return 0.0, -1

    if M > N:
        log.error(
            "Suspect clip (%ds) is longer than the broadcast (%ds). "
            "Match is mathematically impossible.", M, N
        )
        return 0.0, -1

    log.info("Sliding-window search: %d windows × %ds clip inside %ds broadcast …",
             N - M + 1, M, N)

    # ── Vectorised conv1d approach ───────────────────────────────────────────
    # Shape: (1, D, N)
    signal = official_vectors.T.unsqueeze(0).float()
    # Shape: (D, 1, M)  — D separate filters of length M
    kernel = suspect_vectors.T.unsqueeze(1).float()

    # Output: (1, D, N-M+1) — one score per (channel, window) pair
    conv_out = F.conv1d(signal, kernel, groups=D)   # grouped: channel i uses filter i

    # Sum over D channels, normalise by M → mean cosine similarity per window
    window_scores = conv_out.squeeze(0).sum(dim=0) / M   # shape: (N-M+1,)

    best_score_t, best_idx_t = window_scores.max(dim=0)
    best_score = best_score_t.item()
    best_idx   = best_idx_t.item()

    # ── Verdict ──────────────────────────────────────────────────────────────
    log.info("─" * 56)
    if best_score >= threshold:
        log.warning("🚨  VERDICT : PIRACY DETECTED")
        log.warning("    Confidence  : %.4f  (threshold %.2f)", best_score, threshold)
        log.warning("    Location    : ~%d s into official broadcast", best_idx)
    else:
        log.info("✅  VERDICT : CLEAR")
        log.info("    Best score  : %.4f  (below threshold %.2f)", best_score, threshold)
    log.info("─" * 56)

    return best_score, best_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def verify_video_with_sscd(
    official_video_path: str,
    suspect_video_path:  str,
    model_dir:           Path,
    fps_interval:        float = 1.0,
    threshold:           float = DEFAULT_THRESHOLD,
    use_fp16:            bool  = True,
) -> Tuple[float, int]:
    """
    End-to-end SSCD piracy-detection pipeline.

    Parameters
    ----------
    official_video_path : Path to the reference / original broadcast.
    suspect_video_path  : Path to the clip under investigation.
    model_dir           : Directory where the SSCD .torchscript model lives.
    fps_interval        : Seconds between sampled frames (default 1 → 1 fps).
    threshold           : Minimum mean cosine similarity to declare a match.
    use_fp16            : Enable half-precision inference (recommended for 4 GB GPU).

    Returns
    -------
    (score, timestamp)  — timestamp is the start second of the best match,
                          or -1 when no match was found / pipeline failed.
    """
    log.info("=" * 56)
    log.info("  SSCD Video Piracy Detection  —  starting")
    log.info("=" * 56)

    device     = _get_device()
    model_path = ensure_sscd_model_exists(model_dir)
    model      = _load_model(model_path, device, use_fp16)
    batch_size = _auto_batch_size(device, use_fp16)

    try:
        # ── Step 1: Extract frames ────────────────────────────────────────────
        log.info("\n[1/3] Extracting keyframes …")
        official_frames = extract_keyframes(official_video_path, fps_interval)
        suspect_frames  = extract_keyframes(suspect_video_path,  fps_interval)

        if not official_frames:
            raise RuntimeError("No usable frames from official video.")
        if not suspect_frames:
            raise RuntimeError("No usable frames from suspect video.")

        # ── Step 2: Generate embeddings ───────────────────────────────────────
        log.info("\n[2/3] Generating SSCD embeddings …")

        log.info("  Official video …")
        vec_official = get_sscd_vectors(official_frames, model, device, batch_size, use_fp16)
        del official_frames
        gc.collect()
        log.info("  Official vectors: %s", tuple(vec_official.shape))

        log.info("  Suspect video …")
        vec_suspect = get_sscd_vectors(suspect_frames, model, device, batch_size, use_fp16)
        del suspect_frames
        gc.collect()
        log.info("  Suspect  vectors: %s", tuple(vec_suspect.shape))

        # Free the model from VRAM — it is no longer needed for the search step
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        log.info("  Model unloaded from VRAM.")

        # ── Step 3: Sliding-window search ─────────────────────────────────────
        log.info("\n[3/3] Running vectorised sliding-window search …")
        score, timestamp = find_clip_in_long_video(
            vec_official, vec_suspect, threshold=threshold
        )
        return score, timestamp

    except Exception:
        log.exception("Pipeline encountered a fatal error.")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SCRIPT_DIR   = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    MODEL_DIR    = PROJECT_ROOT / "models"

    official_vid = PROJECT_ROOT / "videos" / "official.mp4"
    suspect_vid  = PROJECT_ROOT / "videos" / "official.mp4"

    for p in (official_vid, suspect_vid):
        if not p.exists():
            log.error("Missing video file: %s", p)
            raise SystemExit(1)

    verify_video_with_sscd(
        str(official_vid),
        str(suspect_vid),
        MODEL_DIR,
        fps_interval = 1.0,
        threshold    = DEFAULT_THRESHOLD,
        use_fp16     = True,          # set False if you get numerical anomalies
    )