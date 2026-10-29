"""Generate sample target painting, source brush library, and demonstrate baseline vs adaptive mosaics."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

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


def build_sample_target(output_path: Path) -> Path:
    """Create a rich synthetic target painting with smooth gradients and sharp structures."""
    w, h = 400, 400
    img = Image.new("RGB", (w, h), (20, 25, 45))
    draw = ImageDraw.Draw(img)

    # 1. Sunset sky gradient
    for y in range(180):
        t = y / 180.0
        r = int(240 * (1 - t) + 40 * t)
        g = int(120 * (1 - t) + 60 * t)
        b = int(40 * (1 - t) + 140 * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # 2. Glowing celestial sun / focal orb
    draw.ellipse([240, 50, 320, 130], fill=(255, 235, 140), outline=(255, 140, 20), width=3)

    # 3. Layered mountains with curved silhouettes
    pts1 = [(0, 220), (100, 150), (220, 210), (320, 160), (400, 200), (400, 400), (0, 400)]
    draw.polygon(pts1, fill=(50, 40, 80))

    pts2 = [(0, 260), (140, 200), (280, 270), (400, 230), (400, 400), (0, 400)]
    draw.polygon(pts2, fill=(35, 75, 55))

    # 4. Foreground lake with reflection
    draw.polygon([(0, 310), (400, 310), (400, 400), (0, 400)], fill=(25, 60, 95))
    for rx in range(250, 310, 8):
        draw.line([(rx - 20, 325 + (rx % 5) * 5), (rx + 20, 325 + (rx % 5) * 5)], fill=(240, 180, 100), width=2)

    img.save(output_path)
    return output_path


def build_sample_sources(sources_dir: Path) -> Path:
    """Create a curated set of textured source tiles representing brushstrokes."""
    sources_dir.mkdir(parents=True, exist_ok=True)

    palettes = [
        ("crimson_flame", (210, 40, 50), (255, 150, 50)),
        ("golden_sun", (245, 200, 35), (255, 255, 180)),
        ("emerald_leaf", (35, 150, 70), (100, 230, 120)),
        ("deep_ocean", (30, 70, 180), (100, 180, 240)),
        ("midnight_purple", (80, 35, 130), (160, 90, 220)),
        ("warm_amber", (200, 120, 40), (250, 190, 110)),
        ("soft_cloud", (230, 235, 245), (180, 195, 215)),
        ("obsidian_stone", (35, 35, 42), (80, 85, 95)),
    ]

    for name, c1, c2 in palettes:
        tile = Image.new("RGB", (96, 96), c1)
        draw = ImageDraw.Draw(tile)
        # Add rich multi-frequency internal textures
        for i in range(0, 96, 12):
            draw.line([(0, i), (96, 96 - i)], fill=c2, width=2)
            draw.ellipse([i // 2, i // 2, 96 - i // 2, 96 - i // 2], outline=c2, width=1)
        tile.save(sources_dir / f"{name}.png")

    return sources_dir


def main() -> None:
    examples_dir = Path(__file__).resolve().parent
    target_path = examples_dir / "sample_target.png"
    sources_dir = examples_dir / "sample_sources"
    out_dir = examples_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Building sample target painting...")
    build_sample_target(target_path)
    print(f"  Target painting saved to: {target_path}")

    print("Step 2: Building sample source image collection...")
    build_sample_sources(sources_dir)
    print(f"  Source collection ready at: {sources_dir}")

    # Load source library and analyze painting once
    library = ImageLibrary(sources_dir=sources_dir)
    library.load()
    analyzer = PaintingAnalyzer(image_or_path=target_path)

    # 1. Phase A: Uniform-grid baseline
    print("\nStep 3: Rendering Baseline Grid Photomosaic (Phase A)...")
    grid_config = MosaicConfig(
        target_path=target_path,
        sources_dir=sources_dir,
        output_path=out_dir / "grid_mosaic.png",
        mode=RenderingMode.GRID,
        tile_size=20,
        color_correction=ColorCorrectionMode.REINHARD,
        color_strength=0.85,
        original_blend=0.10,
        seed=42,
    )
    grid_plan = GridStrokePlanner(grid_config).plan(analyzer, library)
    grid_mosaic = MosaicRenderer(grid_config).render(grid_plan, library, analyzer)
    save_output(grid_mosaic, grid_config.output_path)
    grid_metrics = evaluate_quality(analyzer.target_image, grid_mosaic, grid_plan)
    print(f"  Grid mosaic saved to: {grid_config.output_path}")
    print(f"  Total tiles: {grid_metrics.total_strokes}, PSNR: {grid_metrics.psnr_db} dB, Mean Delta E: {grid_metrics.lab_delta_e_mean}")

    # 2. Phases C-F: Composition-Aware Adaptive Brush Mosaic
    print("\nStep 4: Rendering Composition-Aware Adaptive Brush Mosaic (Phases C-F)...")
    adaptive_config = MosaicConfig(
        target_path=target_path,
        sources_dir=sources_dir,
        output_path=out_dir / "adaptive_brush_mosaic.png",
        preview_path=out_dir / "adaptive_brush_mosaic_preview.png",
        mode=RenderingMode.ADAPTIVE,
        min_stroke_size=12,
        max_stroke_size=44,
        stroke_aspect_ratio=1.3,
        brush_mask=BrushMaskType.ORGANIC_STROKE,
        color_correction=ColorCorrectionMode.REINHARD,
        color_strength=0.85,
        luminance_preservation=0.5,
        original_blend=0.15,
        enable_directional=True,
        max_rotation_degrees=75.0,
        overlap_ratio=0.25,
        reuse_penalty_weight=0.3,
        seed=42,
    )
    adaptive_plan = AdaptiveStrokePlanner(adaptive_config).plan(analyzer, library)
    adaptive_mosaic = MosaicRenderer(adaptive_config).render(adaptive_plan, library, analyzer)
    save_output(adaptive_mosaic, adaptive_config.output_path, preview_path=adaptive_config.preview_path)
    adaptive_metrics = evaluate_quality(analyzer.target_image, adaptive_mosaic, adaptive_plan)
    print(f"  Adaptive mosaic saved to: {adaptive_config.output_path}")
    print(f"  Total strokes: {adaptive_metrics.total_strokes}, PSNR: {adaptive_metrics.psnr_db} dB, Mean Delta E: {adaptive_metrics.lab_delta_e_mean}")
    print(f"  Distinct sources: {adaptive_metrics.distinct_sources_used}, Usage entropy: {adaptive_metrics.repetition_entropy} bits")
    print("\nSample generation complete!")


if __name__ == "__main__":
    main()
