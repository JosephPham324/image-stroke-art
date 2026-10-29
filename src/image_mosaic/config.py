"""Configuration models and enumerations for the Image Brush Mosaic Generator."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


class RenderingMode(str, Enum):
    """Rendering approach: uniform grid or spatially-adaptive strokes."""
    GRID = "grid"
    ADAPTIVE = "adaptive"


class BrushMaskType(str, Enum):
    """Mask profile used for individual stroke boundaries."""
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    ORGANIC_STROKE = "organic_stroke"


class ColorCorrectionMode(str, Enum):
    """Method used to adapt source image colors to target region."""
    NONE = "none"
    MEAN_SHIFT = "mean_shift"
    REINHARD = "reinhard"
    TINT = "tint"


@dataclass
class MatchWeights:
    """Weights for the multi-attribute descriptor matching objective."""
    color: float = 1.0
    luminance: float = 0.5
    histogram: float = 0.3
    texture: float = 0.2
    reuse: float = 0.3


@dataclass
class MosaicConfig:
    """Complete rendering configuration for image mosaic generation."""

    target_path: Path
    sources_dir: Path
    output_path: Path
    preview_path: Optional[Path] = None

    # Operational modes
    mode: RenderingMode = RenderingMode.ADAPTIVE
    brush_mask: BrushMaskType = BrushMaskType.ORGANIC_STROKE
    color_correction: ColorCorrectionMode = ColorCorrectionMode.REINHARD

    # Grid parameters
    tile_size: int = 24

    # Adaptive stroke parameters
    min_stroke_size: int = 12
    max_stroke_size: int = 48
    stroke_aspect_ratio: float = 1.2
    stroke_density: float = 1.3
    enable_directional: bool = True
    max_rotation_degrees: float = 90.0
    overlap_ratio: float = 0.25

    # Color & blending adjustments
    color_strength: float = 0.85
    luminance_preservation: float = 0.5
    original_blend: float = 0.15

    # Repetition control
    max_reuse_count: int = 0  # 0 indicates no hard ceiling
    reuse_penalty_weight: float = 0.3

    # Matching objective weights
    weights: MatchWeights = field(default_factory=MatchWeights)

    # Global constraints & determinism
    seed: Optional[int] = 42
    max_resolution: Optional[int] = None
    refinement_passes: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        """Sanitize path attributes against Windows trailing quote / escape artifacts."""
        if self.target_path is not None:
            self.target_path = Path(str(self.target_path).strip().strip('"\''))
        if self.sources_dir is not None:
            self.sources_dir = Path(str(self.sources_dir).strip().strip('"\''))
        if self.output_path is not None:
            self.output_path = Path(str(self.output_path).strip().strip('"\''))
        if self.preview_path is not None:
            self.preview_path = Path(str(self.preview_path).strip().strip('"\''))

    def validate(self) -> None:
        """Validate configuration settings and raise ValueError on invalid values."""
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target image not found: {self.target_path}")
        if not self.sources_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.sources_dir}")
        if not self.sources_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {self.sources_dir}")

        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.min_stroke_size <= 0:
            raise ValueError(f"min_stroke_size must be positive, got {self.min_stroke_size}")
        if self.max_stroke_size < self.min_stroke_size:
            raise ValueError(
                f"max_stroke_size ({self.max_stroke_size}) cannot be smaller than min_stroke_size ({self.min_stroke_size})"
            )
        if self.stroke_aspect_ratio <= 0:
            raise ValueError(f"stroke_aspect_ratio must be positive, got {self.stroke_aspect_ratio}")

        if not 0.0 <= self.color_strength <= 1.0:
            raise ValueError(f"color_strength must be in [0.0, 1.0], got {self.color_strength}")
        if not 0.0 <= self.original_blend <= 1.0:
            raise ValueError(f"original_blend must be in [0.0, 1.0], got {self.original_blend}")
        if not 0.0 <= self.luminance_preservation <= 1.0:
            raise ValueError(
                f"luminance_preservation must be in [0.0, 1.0], got {self.luminance_preservation}"
            )
        if self.max_resolution is not None and self.max_resolution <= 0:
            raise ValueError(f"max_resolution must be positive, got {self.max_resolution}")
