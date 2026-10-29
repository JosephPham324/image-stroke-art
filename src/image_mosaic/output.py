"""Output generation, preview export, and composition metric evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from image_mosaic.stroke_planner import StrokePlan


@dataclass
class QualityMetrics:
    """Quantitative diagnostics of reconstruction fidelity and library utilization."""
    psnr_db: float
    lab_delta_e_mean: float
    total_strokes: int
    distinct_sources_used: int
    max_source_reuse: int
    repetition_entropy: float


def evaluate_quality(
    target_img: Image.Image,
    mosaic_img: Image.Image,
    plan: Optional[StrokePlan] = None,
) -> QualityMetrics:
    """Compute perceptual color and composition reconstruction metrics."""
    w, h = mosaic_img.size
    tgt_resized = target_img.resize((w, h), Image.Resampling.BILINEAR)

    tgt_arr = np.array(tgt_resized, dtype=np.float32) / 255.0
    mos_arr = np.array(mosaic_img, dtype=np.float32) / 255.0

    # MSE and PSNR in RGB
    mse = float(np.mean((tgt_arr - mos_arr) ** 2))
    if mse < 1e-10:
        psnr = 99.0
    else:
        psnr = float(10.0 * math.log10(1.0 / mse))

    # Delta E in CIELAB
    tgt_lab = cv2.cvtColor(tgt_arr, cv2.COLOR_RGB2LAB)
    mos_lab = cv2.cvtColor(mos_arr, cv2.COLOR_RGB2LAB)
    delta_e = np.sqrt(np.sum((tgt_lab - mos_lab) ** 2, axis=2))
    mean_delta_e = float(np.mean(delta_e))

    # Stroke statistics
    total_strokes = plan.total_strokes if plan else 0
    if plan and plan.strokes:
        source_counts: Dict[int, int] = {}
        for s in plan.strokes:
            source_counts[s.source_idx] = source_counts.get(s.source_idx, 0) + 1

        distinct_used = len(source_counts)
        max_reuse = max(source_counts.values())

        # Shannon entropy of source usage: H = -sum(p * log2(p))
        probs = np.array(list(source_counts.values()), dtype=np.float64) / total_strokes
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        distinct_used = 0
        max_reuse = 0
        entropy = 0.0

    return QualityMetrics(
        psnr_db=round(psnr, 2),
        lab_delta_e_mean=round(mean_delta_e, 2),
        total_strokes=total_strokes,
        distinct_sources_used=distinct_used,
        max_source_reuse=max_reuse,
        repetition_entropy=round(entropy, 2),
    )


def save_output(
    image: Image.Image,
    output_path: Path,
    preview_path: Optional[Path] = None,
    preview_max_dim: int = 512,
) -> None:
    """Save rendered mosaic image and optional downscaled preview."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save primary image
    image.save(output_path)

    # Save preview if requested
    if preview_path:
        preview_path = Path(preview_path)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = image.size
        scale = min(1.0, float(preview_max_dim) / max(w, h))
        pw = max(1, int(w * scale))
        ph = max(1, int(h * scale))
        preview = image.resize((pw, ph), Image.Resampling.BILINEAR)
        preview.save(preview_path)
