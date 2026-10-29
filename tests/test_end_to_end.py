"""End-to-end integration tests for image brush mosaic generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from image_mosaic.config import (
    BrushMaskType,
    ColorCorrectionMode,
    MosaicConfig,
    RenderingMode,
)
from image_mosaic.image_library import ImageLibrary
from image_mosaic.output import evaluate_quality, save_output
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.renderer import MosaicRenderer
from image_mosaic.stroke_planner import AdaptiveStrokePlanner, GridStrokePlanner


def test_end_to_end_grid_mode(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    out_path = tmp_path / "mosaic_grid.png"
    preview_path = tmp_path / "mosaic_grid_preview.png"

    config = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=out_path,
        preview_path=preview_path,
        mode=RenderingMode.GRID,
        tile_size=25,
        color_correction=ColorCorrectionMode.REINHARD,
        color_strength=0.8,
        original_blend=0.1,
        seed=42,
    )

    library = ImageLibrary(config.sources_dir)
    library.load()
    analyzer = PaintingAnalyzer(config.target_path)

    plan = GridStrokePlanner(config).plan(analyzer, library)
    renderer = MosaicRenderer(config)
    result = renderer.render(plan, library, analyzer)

    save_output(result, out_path, preview_path)
    metrics = evaluate_quality(analyzer.target_image, result, plan)

    # Verifications
    assert out_path.exists()
    assert preview_path.exists()
    assert result.size == (200, 200)

    # Check metrics
    assert metrics.total_strokes == 64  # (200/25)^2 = 8*8 = 64 tiles
    assert metrics.distinct_sources_used > 1
    assert metrics.psnr_db > 10.0
    assert metrics.lab_delta_e_mean < 40.0


def test_end_to_end_adaptive_mode(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    out_path = tmp_path / "mosaic_adaptive.png"
    preview_path = tmp_path / "mosaic_adaptive_preview.png"

    config = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=out_path,
        preview_path=preview_path,
        mode=RenderingMode.ADAPTIVE,
        min_stroke_size=12,
        max_stroke_size=36,
        brush_mask=BrushMaskType.ORGANIC_STROKE,
        color_correction=ColorCorrectionMode.REINHARD,
        color_strength=0.85,
        original_blend=0.15,
        enable_directional=True,
        seed=42,
    )

    library = ImageLibrary(config.sources_dir)
    library.load()
    analyzer = PaintingAnalyzer(config.target_path)

    plan = AdaptiveStrokePlanner(config).plan(analyzer, library)
    renderer = MosaicRenderer(config)
    result = renderer.render(plan, library, analyzer)

    save_output(result, out_path, preview_path)
    metrics = evaluate_quality(analyzer.target_image, result, plan)

    assert out_path.exists()
    assert preview_path.exists()
    assert result.size == (200, 200)

    # Ensure strokes were planned across multiple layers and directional angles
    angles = [s.angle for s in plan.strokes]
    assert any(abs(a) > 5.0 for a in angles), "Expected directional strokes to have non-zero rotation"

    assert metrics.total_strokes > 30
    assert metrics.distinct_sources_used >= 3
    assert metrics.repetition_entropy > 0.5
