"""Composition-Aware Image Brush Mosaic Generator.

Reconstructs target paintings or photographs using a collection of source images
as irregular, directional, and composition-aware visual brushstrokes.
"""

from image_mosaic.config import (
    BrushMaskType,
    ColorCorrectionMode,
    MosaicConfig,
    RenderingMode,
)
from image_mosaic.descriptors import DescriptorExtractor, MatchScoreWeights
from image_mosaic.image_library import ImageLibrary, SourceImage
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.renderer import MosaicRenderer
from image_mosaic.stroke_planner import AdaptiveStrokePlanner, GridStrokePlanner, Stroke, StrokePlan

__version__ = "0.1.0"

__all__ = [
    "BrushMaskType",
    "ColorCorrectionMode",
    "MosaicConfig",
    "RenderingMode",
    "DescriptorExtractor",
    "MatchScoreWeights",
    "ImageLibrary",
    "SourceImage",
    "PaintingAnalyzer",
    "MosaicRenderer",
    "GridStrokePlanner",
    "AdaptiveStrokePlanner",
    "Stroke",
    "StrokePlan",
]
