"""Tests for grid and adaptive stroke planners."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from image_mosaic.config import MosaicConfig, RenderingMode
from image_mosaic.image_library import ImageLibrary
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.stroke_planner import AdaptiveStrokePlanner, GridStrokePlanner


def test_grid_stroke_planner_tile_count(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    config = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=tmp_path / "out.png",
        mode=RenderingMode.GRID,
        tile_size=40,
    )
    lib = ImageLibrary(synthetic_sources_dir)
    lib.load()
    analyzer = PaintingAnalyzer(synthetic_target)

    planner = GridStrokePlanner(config)
    plan = planner.plan(analyzer, lib)

    # 200 / 40 = 5 rows, 5 cols -> 25 tiles
    assert plan.total_strokes == 25
    assert plan.canvas_width == 200
    assert plan.canvas_height == 200


def test_adaptive_stroke_planner_determinism(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    config1 = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=tmp_path / "out1.png",
        mode=RenderingMode.ADAPTIVE,
        min_stroke_size=16,
        max_stroke_size=32,
        seed=123,
    )
    config2 = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=tmp_path / "out2.png",
        mode=RenderingMode.ADAPTIVE,
        min_stroke_size=16,
        max_stroke_size=32,
        seed=123,
    )
    lib = ImageLibrary(synthetic_sources_dir)
    lib.load()
    analyzer = PaintingAnalyzer(synthetic_target)

    plan1 = AdaptiveStrokePlanner(config1).plan(analyzer, lib)
    plan2 = AdaptiveStrokePlanner(config2).plan(analyzer, lib)

    assert plan1.total_strokes == plan2.total_strokes
    for s1, s2 in zip(plan1.strokes, plan2.strokes):
        assert pytest.approx(s1.x, abs=1e-4) == s2.x
        assert pytest.approx(s1.y, abs=1e-4) == s2.y
        assert s1.source_idx == s2.source_idx
        assert pytest.approx(s1.angle, abs=1e-4) == s2.angle


def test_adaptive_strokes_vary_in_size(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    config = MosaicConfig(
        target_path=synthetic_target,
        sources_dir=synthetic_sources_dir,
        output_path=tmp_path / "out.png",
        mode=RenderingMode.ADAPTIVE,
        min_stroke_size=12,
        max_stroke_size=40,
        seed=42,
    )
    lib = ImageLibrary(synthetic_sources_dir)
    lib.load()
    analyzer = PaintingAnalyzer(synthetic_target)

    plan = AdaptiveStrokePlanner(config).plan(analyzer, lib)
    sizes = [s.height for s in plan.strokes]

    # Verify that we have multiple stroke scales
    assert min(sizes) < max(sizes)
    assert min(sizes) <= 20
    assert max(sizes) >= 30
