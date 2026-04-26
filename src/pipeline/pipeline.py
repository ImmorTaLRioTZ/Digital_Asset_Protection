"""
Detection Pipeline — Orchestrator
===================================
Chains Stage 1 (pHash), Stage 2 (Audio Fingerprint), and Stage 3 (SSCD)
into a single, efficient cascade. Each stage only runs when the previous
one fails to find a match, conserving CPU/GPU resources.

Usage
-----
    from pipeline import DetectionPipeline, PipelineConfig
    from pathlib import Path

    # 1. Build your reference database (once, then persist / cache)
    pipeline = DetectionPipeline(PipelineConfig(sscd_model_dir=Path("models/")))
    pipeline.register_asset("my_video.mp4", asset_id="asset_001")

    # 2. Check a suspicious video
    result = pipeline.check("suspect.mp4")
    if result.is_pirated:
        print(f"PIRATED — matched '{result.asset_id}' via {result.detected_by}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import src.pipeline.stage1_phash as phash_stage
import src.pipeline.stage2_audiomapping as audio_stage
import src.pipeline.stage3_sscd  as sscd_stage

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Final verdict returned for every checked video."""
    is_pirated: bool
    asset_id: Optional[str]     = None
    detected_by: Optional[str]  = None   # "Stage1_pHash" | "Stage2_Audio" | "Stage3_SSCD"
    confidence: str              = "low"  # "high" | "medium" | "low"
    detail: str                  = ""

    def __str__(self) -> str:
        status = "⛔ PIRATED" if self.is_pirated else "✅ CLEAN"
        if self.is_pirated:
            return (
                f"{status} — asset='{self.asset_id}' "
                f"detected_by={self.detected_by} "
                f"confidence={self.confidence}\n  {self.detail}"
            )
        return f"{status} — {self.detail}"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Stage 1 — pHash
    phash_hamming_threshold: int   = phash_stage.HAMMING_THRESHOLD
    phash_match_ratio: float       = phash_stage.MATCH_FRAME_RATIO

    # Stage 2 — Audio fingerprinting
    audio_vote_threshold: int      = audio_stage.VOTE_THRESHOLD

    # Stage 3 — SSCD
    # Set sscd_model_dir to enable Stage 3; leave None to skip it entirely
    # (videos that defeat Stages 1 and 2 will be marked CLEAN).
    sscd_model_dir: Optional[Path] = None
    sscd_threshold: float          = sscd_stage.DEFAULT_THRESHOLD
    sscd_fps_interval: float       = sscd_stage.FPS_INTERVAL
    sscd_use_fp16: bool            = True


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DetectionPipeline:
    """
    Three-stage cascade:
        Stage 1 (CPU, ~ms)   → pHash frame similarity
        Stage 2 (CPU, ~s)    → Audio landmark fingerprinting
        Stage 3 (GPU, ~10 s) → SSCD visual embedding
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._phash_db: list[phash_stage.VideoFingerprint] = []
        self._audio_db: list[audio_stage.AudioFingerprint] = []
        self._sscd_db:  list[sscd_stage.VideoFingerprint]  = []

    # ── Registration ──────────────────────────────────────────────────────────

    def register_asset(
        self,
        video_path:  str | Path,
        asset_id:    str,
        skip_audio:  bool = False,
        skip_sscd:   bool = False,
    ) -> None:
        """
        Fingerprint an official reference video and add it to all databases.

        Stages 1 and 2 are always registered.
        Stage 3 is registered only when `config.sscd_model_dir` is set and
        `skip_sscd` is False. The model is loaded, used, and unloaded for
        each call — VRAM is never held between registrations.
        """
        path = Path(video_path)
        logger.info("Registering asset '%s' from %s …", asset_id, path.name)

        # Stage 1 fingerprint
        pf = phash_stage.build_fingerprint(path, asset_id)
        self._phash_db.append(pf)

        # Stage 2 fingerprint
        if not skip_audio:
            af = audio_stage.build_fingerprint(path, asset_id)
            self._audio_db.append(af)

        # Stage 3 fingerprint
        if not skip_sscd and self.config.sscd_model_dir is not None:
            sf = sscd_stage.build_fingerprint(
                path,
                asset_id,
                model_dir=self.config.sscd_model_dir,
                fps_interval=self.config.sscd_fps_interval,
                use_fp16=self.config.sscd_use_fp16,
            )
            self._sscd_db.append(sf)

        logger.info(
            "Asset '%s' registered — pHash: %d  audio: %s  sscd: %s",
            asset_id,
            len(pf.frame_hashes),
            f"{len(self._audio_db[-1].hashes)} hashes" if not skip_audio else "skipped",
            f"{self._sscd_db[-1].frame_count} vectors"
            if (not skip_sscd and self.config.sscd_model_dir) else "skipped",
        )

    # ── Manual fingerprint loaders (for pre-computed / cached fingerprints) ───

    def load_phash_fingerprint(self, fp: phash_stage.VideoFingerprint) -> None:
        """Add a pre-computed pHash fingerprint (e.g. loaded from DB/cache)."""
        self._phash_db.append(fp)

    def load_audio_fingerprint(self, fp: audio_stage.AudioFingerprint) -> None:
        """Add a pre-computed audio fingerprint (e.g. loaded from DB/cache)."""
        self._audio_db.append(fp)

    def load_sscd_fingerprint(self, fp: sscd_stage.VideoFingerprint) -> None:
        """Add a pre-computed SSCD fingerprint (e.g. loaded from DB/cache)."""
        self._sscd_db.append(fp)

    # ── Main check ────────────────────────────────────────────────────────────

    def check(self, video_path: str | Path) -> PipelineResult:
        """
        Run the full cascade against *video_path*.

        Stops at the earliest stage that finds a match. Stage 3 is only
        invoked when Stages 1 and 2 both pass, keeping the GPU idle for the
        vast majority of pirated uploads.
        """
        path = Path(video_path)
        logger.info("━" * 60)
        logger.info("Checking: %s", path.name)

        # ── Stage 1: pHash ────────────────────────────────────────────────────
        if self._phash_db:
            s1 = phash_stage.check_video(
                path,
                self._phash_db,
                hamming_threshold=self.config.phash_hamming_threshold,
                match_ratio_threshold=self.config.phash_match_ratio,
            )
            if s1.matched:
                return PipelineResult(
                    is_pirated=True,
                    asset_id=s1.asset_id,
                    detected_by="Stage1_pHash",
                    confidence="high",
                    detail=s1.reason,
                )
        else:
            logger.warning("pHash database is empty — skipping Stage 1")

        # ── Stage 2: Audio fingerprinting ─────────────────────────────────────
        if self._audio_db:
            s2 = audio_stage.check_video(
                path,
                self._audio_db,
                vote_threshold=self.config.audio_vote_threshold,
            )
            if s2.matched:
                return PipelineResult(
                    is_pirated=True,
                    asset_id=s2.asset_id,
                    detected_by="Stage2_Audio",
                    confidence="high",
                    detail=s2.reason,
                )
        else:
            logger.warning("Audio database is empty — skipping Stage 2")

        # ── Stage 3: SSCD ─────────────────────────────────────────────────────
        if self._sscd_db and self.config.sscd_model_dir is not None:
            s3 = sscd_stage.check_video(
                path,
                self._sscd_db,
                model_dir=self.config.sscd_model_dir,
                threshold=self.config.sscd_threshold,
                fps_interval=self.config.sscd_fps_interval,
                use_fp16=self.config.sscd_use_fp16,
            )
            if s3.matched:
                return PipelineResult(
                    is_pirated=True,
                    asset_id=s3.asset_id,
                    detected_by="Stage3_SSCD",
                    confidence="medium",   # SSCD is similarity-based, not binary
                    detail=s3.reason,
                )
        else:
            logger.info(
                "[Stage 3] %s — pipeline ends here",
                "SSCD database is empty" if not self._sscd_db else "sscd_model_dir not set",
            )

        # ── All stages passed: video is clean ─────────────────────────────────
        return PipelineResult(
            is_pirated=False,
            detail="Passed all active detection stages",
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def registered_assets(self) -> list[str]:
        return [fp.asset_id for fp in self._phash_db]

    def __repr__(self) -> str:
        return (
            f"DetectionPipeline("
            f"phash_db={len(self._phash_db)}, "
            f"audio_db={len(self._audio_db)}, "
            f"sscd_db={len(self._sscd_db)})"
        )