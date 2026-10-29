"""Tests for renderer, masking, color correction, and refinement."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from image_mosaic.config import BrushMaskType, ColorCorrectionMode, MosaicConfig
from image_mosaic.image_library import ImageLibrary
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.renderer import MosaicRenderer
from image_mosaic.stroke_planner import Stroke, StrokePlan


def test_brush_mask_shapes():
    config = MosaicConfig(
        target_path=Path("dummy.png"),
        sources_dir=Path("dummy_dir"),
        output_path=Path("dummy_out.png"),
    )
    renderer = MosaicRenderer(config)

    for mask_type in BrushMaskType:
        mask = renderer._generate_brush_mask(32, 24, mask_type)
        assert mask.size == (32, 24)
        mask_np = np.array(mask)
        assert mask_np.min() >= 0
        assert mask_np.max() <= 255


def test_color_correction_reinhard_no_overflow():
    config = MosaicConfig(
        target_path=Path("dummy.png"),
        sources_dir=Path("dummy_dir"),
        output_path=Path("dummy_out.png"),
    )
    renderer = MosaicRenderer(config)

    src_img = Image.new("RGB", (32, 32), (50, 100, 150))
    target_lab = np.array([80.0, -20.0, 30.0], dtype=np.float32)

    corrected = renderer._apply_color_correction(
        src_rgb=src_img,
        target_lab=target_lab,
        mode=ColorCorrectionMode.REINHARD,
        strength=0.9,
        lum_preservation=0.5,
    )

    arr = np.array(corrected)
    assert arr.shape == (32, 32, 3)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_renderer_boundary_strokes_dont_crash(synthetic_sources_dir: Path, synthetic_target: Path, tmp_path: Path):
    config = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=tmp_path / "out.png",
    )
    lib = ImageLibrary(synthetic_sources_dir)
    lib.load()
    analyzer = PaintingAnalyzer(synthetic_target)

    # Place strokes extending way outside the canvas bounds
    plan = StrokePlan(
        strokes=[
            Stroke(x=-10.0, y=-10.0, width=40.0, height=40.0, angle=45.0, source_idx=0, target_color_lab=np.array([50.0, 0.0, 0.0], dtype=np.float32)),
            Stroke(x=210.0, y=210.0, width=40.0, height=40.0, angle=-30.0, source_idx=1, target_color_lab=np.array([50.0, 0.0, 0.0], dtype=np.float32)),
        ],
        canvas_width=200,
        canvas_height=200,
    )

    renderer = MosaicRenderer(config)
    result = renderer.render(plan, lib, analyzer)
    assert result.size == (200, 200)
