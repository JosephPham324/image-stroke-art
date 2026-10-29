"""Stroke planning: grid-based uniform layout and adaptive multi-scale brush layout."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from image_mosaic.config import MosaicConfig, RenderingMode
from image_mosaic.descriptors import compute_vectorized_match_scores
from image_mosaic.image_library import ImageLibrary
from image_mosaic.painting_analyzer import PaintingAnalyzer


@dataclass
class Stroke:
    """Specification for placing a single source image on the canvas."""
    x: float
    y: float
    width: float
    height: float
    angle: float  # In degrees, 0.0 = horizontal
    source_idx: int
    target_color_lab: np.ndarray  # (3,) float32 CIELAB mean of target region
    layer: int = 0
    importance: float = 0.5


@dataclass
class StrokePlan:
    """Complete collection of strokes ready for rendering."""
    strokes: List[Stroke]
    canvas_width: int
    canvas_height: int

    @property
    def total_strokes(self) -> int:
        return len(self.strokes)


class GridStrokePlanner:
    """Plans regular uniform-grid tiles covering the target image."""

    def __init__(self, config: MosaicConfig) -> None:
        self.config = config

    def plan(
        self,
        analyzer: PaintingAnalyzer,
        library: ImageLibrary,
    ) -> StrokePlan:
        """Generate a uniform grid stroke plan."""
        w, h = analyzer.width, analyzer.height
        tile_size = self.config.tile_size
        cols = math.ceil(w / tile_size)
        rows = math.ceil(h / tile_size)

        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        reuse_counts = np.zeros(library.count, dtype=np.int32)
        strokes: List[Stroke] = []
        source_matrix = library.descriptors_matrix

        for r in range(rows):
            for c in range(cols):
                x0 = c * tile_size
                y0 = r * tile_size
                x1 = min(w, x0 + tile_size)
                y1 = min(h, y0 + tile_size)

                # Center of tile
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                tile_w = x1 - x0
                tile_h = y1 - y0

                # Extract patch descriptor
                patch_desc = analyzer.get_patch_descriptor(x0, y0, x1, y1)

                # Vectorized matching against all source images
                scores = compute_vectorized_match_scores(
                    target_desc=patch_desc,
                    source_matrix=source_matrix,
                    weights=self.config.weights,
                    reuse_counts=reuse_counts,
                    max_reuse=self.config.max_reuse_count,
                )

                best_idx = int(np.argmin(scores))
                reuse_counts[best_idx] += 1

                strokes.append(
                    Stroke(
                        x=cx,
                        y=cy,
                        width=float(tile_w),
                        height=float(tile_h),
                        angle=0.0,
                        source_idx=best_idx,
                        target_color_lab=patch_desc.mean_lab.copy(),
                        layer=0,
                        importance=0.5,
                    )
                )

        return StrokePlan(strokes=strokes, canvas_width=w, canvas_height=h)


class AdaptiveStrokePlanner:
    """Plans multi-scale, directional, irregular strokes guided by target composition and saliency."""

    def __init__(self, config: MosaicConfig) -> None:
        self.config = config

    def plan(
        self,
        analyzer: PaintingAnalyzer,
        library: ImageLibrary,
    ) -> StrokePlan:
        """Generate an adaptive multi-scale stroke plan.
        
        Strategy:
          1. Coarse base layer: Large strokes across broad, smooth regions ensuring complete coverage.
          2. Intermediate layer: Medium strokes in areas with moderate contrast/texture.
          3. Fine detail layer: Small, dense strokes aligned with edge tangents in high-saliency areas.
        """
        w, h = analyzer.width, analyzer.height
        min_size = self.config.min_stroke_size
        max_size = self.config.max_stroke_size
        aspect = self.config.stroke_aspect_ratio
        overlap = self.config.overlap_ratio

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        reuse_counts = np.zeros(library.count, dtype=np.int32)
        source_matrix = library.descriptors_matrix
        strokes: List[Stroke] = []

        # Multi-scale layer definitions: [layer_idx, stroke_size, min_saliency, max_saliency, step_mult]
        # Layer 0: Coarse background (covers canvas with large strokes)
        # Layer 1: Medium features (captures mid-frequency transitions)
        # Layer 2: Fine detail and sharp boundaries (captures edges, faces, fine lines)
        mid_size = max(min_size + 4, int((min_size + max_size) / 2))
        layers = [
            {"layer": 0, "size": max_size, "min_saliency": 0.0, "max_saliency": 0.60, "step_mult": 1.0 - overlap},
            {"layer": 1, "size": mid_size, "min_saliency": 0.15, "max_saliency": 1.0, "step_mult": 0.80 - overlap},
            {"layer": 2, "size": min_size, "min_saliency": 0.25, "max_saliency": 1.0, "step_mult": 0.70 - overlap},
        ]

        for layer_info in layers:
            layer_idx = layer_info["layer"]
            base_size = layer_info["size"]
            min_sal = layer_info["min_saliency"]
            max_sal = layer_info["max_saliency"]
            step = max(4, int(base_size * layer_info["step_mult"] / self.config.stroke_density))

            # Grid sampling with slight deterministic pseudo-random jitter for natural brush feel
            y_coords = list(range(step // 2, h, step))
            x_coords = list(range(step // 2, w, step))

            for y in y_coords:
                for x in x_coords:
                    sal = analyzer.get_saliency_at(x, y, radius=base_size / 2.0)

                    # Only place strokes where saliency matches this layer's scope
                    # (Layer 0 places everywhere up to max_sal to ensure 100% canvas coverage)
                    if layer_idx > 0 and (sal < min_sal or sal > max_sal):
                        continue
                    if layer_idx == 0 and sal > max_sal and len(layers) > 1:
                        # In high-detail areas, coarse layer still places a stroke or leaves to layer 1/2
                        # We place a slightly smaller stroke so underpainting exists
                        pass

                    # Small jitter for organic brushstroke layout
                    jitter_x = (random.random() - 0.5) * (step * 0.3)
                    jitter_y = (random.random() - 0.5) * (step * 0.3)
                    cx = float(np.clip(x + jitter_x, 0, w))
                    cy = float(np.clip(y + jitter_y, 0, h))

                    # Local size scaled subtly by inverse saliency
                    size_mod = base_size * (1.1 - 0.2 * sal)
                    stroke_w = max(4.0, size_mod * aspect)
                    stroke_h = max(4.0, size_mod)

                    # Compute stroke orientation
                    angle = 0.0
                    if self.config.enable_directional:
                        tangent_deg, edge_strength = analyzer.get_tangent_angle_degrees(
                            cx, cy, radius=stroke_h / 2.0
                        )
                        if edge_strength > 0.05:
                            # Align stroke along edge contour
                            angle = float(np.clip(
                                tangent_deg,
                                -self.config.max_rotation_degrees,
                                self.config.max_rotation_degrees
                            ))

                    # Region bounding box for target descriptor
                    half_w = stroke_w / 2.0
                    half_h = stroke_h / 2.0
                    rx0 = int(max(0, cx - half_w))
                    ry0 = int(max(0, cy - half_h))
                    rx1 = int(min(w, cx + half_w))
                    ry1 = int(min(h, cy + half_h))

                    patch_desc = analyzer.get_patch_descriptor(rx0, ry0, rx1, ry1)

                    # Match best candidate from library
                    scores = compute_vectorized_match_scores(
                        target_desc=patch_desc,
                        source_matrix=source_matrix,
                        weights=self.config.weights,
                        reuse_counts=reuse_counts,
                        max_reuse=self.config.max_reuse_count,
                    )

                    best_idx = int(np.argmin(scores))
                    reuse_counts[best_idx] += 1

                    strokes.append(
                        Stroke(
                            x=cx,
                            y=cy,
                            width=stroke_w,
                            height=stroke_h,
                            angle=angle,
                            source_idx=best_idx,
                            target_color_lab=patch_desc.mean_lab.copy(),
                            layer=layer_idx,
                            importance=sal,
                        )
                    )

        # Sort strokes by layer so background renders first, fine details render on top
        strokes.sort(key=lambda s: (s.layer, s.importance))

        return StrokePlan(strokes=strokes, canvas_width=w, canvas_height=h)
