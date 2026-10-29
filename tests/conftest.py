"""Pytest fixtures for synthetic image generation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def synthetic_target(tmp_path: Path) -> Path:
    """Generate a synthetic target painting with color gradients, shapes, and edges."""
    target_path = tmp_path / "target_painting.png"
    w, h = 200, 200
    img = Image.new("RGB", (w, h), (30, 40, 70))
    draw = ImageDraw.Draw(img)

    # Sky gradient / smooth region
    for y in range(80):
        color = (30 + y, 60 + y // 2, 120 + y)
        draw.line([(0, y), (w, y)], fill=color)

    # Bright sun / circular focal point (high contrast edge)
    draw.ellipse([120, 20, 170, 70], fill=(255, 220, 50), outline=(255, 100, 0))

    # Rolling green hills (tangent curvature)
    draw.polygon([(0, 120), (80, 90), (160, 130), (200, 110), (200, 200), (0, 200)], fill=(34, 139, 34))

    # Bright house/box with sharp edges and dark roof
    draw.rectangle([50, 130, 100, 180], fill=(220, 50, 40))
    draw.polygon([(40, 130), (75, 95), (110, 130)], fill=(40, 40, 40))

    img.save(target_path)
    return target_path


@pytest.fixture
def synthetic_sources_dir(tmp_path: Path) -> Path:
    """Create a directory with a variety of textured synthetic source images."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        ("red_tex", (220, 40, 40)),
        ("blue_tex", (40, 80, 220)),
        ("green_tex", (40, 180, 40)),
        ("yellow_tex", (240, 220, 30)),
        ("cyan_tex", (30, 210, 220)),
        ("magenta_tex", (210, 30, 210)),
        ("white_tex", (240, 240, 240)),
        ("dark_tex", (25, 25, 30)),
    ]

    for name, base_color in colors:
        img = Image.new("RGB", (64, 64), base_color)
        draw = ImageDraw.Draw(img)
        # Add internal texture/pattern
        draw.line([(0, 0), (64, 64)], fill=(255, 255, 255), width=3)
        draw.ellipse([16, 16, 48, 48], outline=(0, 0, 0), width=2)
        img.save(sources_dir / f"{name}.png")

    # Add a transparent RGBA image
    rgba_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw_rgba = ImageDraw.Draw(rgba_img)
    draw_rgba.ellipse([10, 10, 54, 54], fill=(255, 128, 0, 230), outline=(255, 255, 255, 255))
    rgba_img.save(sources_dir / "transparent_face.png")

    # Add an exact duplicate file
    dup_img = Image.open(sources_dir / "red_tex.png")
    dup_img.save(sources_dir / "red_tex_duplicate.png")

    # Add a corrupted non-image file with .jpg extension
    corrupt_file = sources_dir / "corrupted.jpg"
    corrupt_file.write_bytes(b"CORRUPTED_NOT_AN_IMAGE_DATA_12345")

    return sources_dir
