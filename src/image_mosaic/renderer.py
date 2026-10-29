"""Renderer: translates stroke plans into composite visual mosaics."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from image_mosaic.config import BrushMaskType, ColorCorrectionMode, MosaicConfig
from image_mosaic.image_library import ImageLibrary
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.stroke_planner import Stroke, StrokePlan


class MosaicRenderer:
    """Renders planned strokes onto an output canvas with masking, color correction, and refinement."""

    def __init__(self, config: MosaicConfig) -> None:
        self.config = config

    def render(
        self,
        plan: StrokePlan,
        library: ImageLibrary,
        analyzer: PaintingAnalyzer,
    ) -> Image.Image:
        """Execute the stroke plan and produce the rendered mosaic image."""
        w, h = plan.canvas_width, plan.canvas_height

        # Start with a canvas filled with neutral background color from target average
        target_avg_rgb = tuple(int(round(c * 255.0)) for c in np.mean(analyzer.rgb, axis=(0, 1)))
        canvas = Image.new("RGBA", (w, h), (*target_avg_rgb, 255))

        # Render each planned stroke
        for stroke in plan.strokes:
            self._render_single_stroke(canvas, stroke, library)

        # Refinement passes (Phase E & F)
        result_img = self._apply_refinements(canvas.convert("RGB"), analyzer)

        return result_img

    def _render_single_stroke(
        self,
        canvas: Image.Image,
        stroke: Stroke,
        library: ImageLibrary,
    ) -> None:
        """Transform, mask, color-correct, and alpha-composite a single stroke."""
        cw, ch = canvas.size
        sw = max(2, int(round(stroke.width)))
        sh = max(2, int(round(stroke.height)))

        # Retrieve source image sized to stroke dimensions
        src_img = library.get_source_pil(stroke.source_idx, target_size=(sw, sh))

        # Separate color channels and existing alpha
        if src_img.mode == "RGBA":
            src_rgb = src_img.convert("RGB")
            src_alpha = src_img.split()[3]
        else:
            src_rgb = src_img.convert("RGB")
            src_alpha = Image.new("L", (sw, sh), 255)

        # Apply color & luminance adaptation to RGB channels
        corrected_rgb = self._apply_color_correction(
            src_rgb=src_rgb,
            target_lab=stroke.target_color_lab,
            mode=self.config.color_correction,
            strength=self.config.color_strength,
            lum_preservation=self.config.luminance_preservation,
        )

        # Generate stroke mask
        mask = self._generate_brush_mask(
            width=sw,
            height=sh,
            mask_type=self.config.brush_mask,
        )

        # Combine mask with source image's intrinsic alpha
        combined_alpha = ImageOps.invert(
            ImageOps.invert(src_alpha)  # preserves full opacity where both are opaque
        )
        combined_alpha_np = (
            np.array(src_alpha, dtype=np.float32)
            * np.array(mask, dtype=np.float32)
            / 255.0
        ).astype(np.uint8)
        combined_alpha = Image.fromarray(combined_alpha_np, mode="L")

        # Combine into RGBA stroke tile
        stroke_rgba = Image.new("RGBA", (sw, sh))
        stroke_rgba.paste(corrected_rgb, (0, 0))
        stroke_rgba.putalpha(combined_alpha)

        # Apply rotation if specified
        if abs(stroke.angle) > 0.5:
            # Rotate with bilinear filtering and expanding canvas to fit rotated bounds
            rotated_rgba = stroke_rgba.rotate(
                stroke.angle,
                resample=Image.Resampling.BILINEAR,
                expand=True,
            )
        else:
            rotated_rgba = stroke_rgba

        rw, rh = rotated_rgba.size
        px = int(round(stroke.x - rw / 2.0))
        py = int(round(stroke.y - rh / 2.0))

        # Check canvas boundary intersection
        cx0 = max(0, px)
        cy0 = max(0, py)
        cx1 = min(cw, px + rw)
        cy1 = min(ch, py + rh)

        if cx1 <= cx0 or cy1 <= cy0:
            return

        # Crop rotated patch to visible canvas rectangle
        px0 = cx0 - px
        py0 = cy0 - py
        px1 = px0 + (cx1 - cx0)
        py1 = py0 + (cy1 - cy0)

        cropped_patch = rotated_rgba.crop((px0, py0, px1, py1))
        canvas.alpha_composite(cropped_patch, (cx0, cy0))

    def _generate_brush_mask(
        self,
        width: int,
        height: int,
        mask_type: BrushMaskType,
    ) -> Image.Image:
        """Create an antialiased 8-bit alpha mask for the brush shape."""
        if mask_type == BrushMaskType.RECTANGLE:
            # Soft rectangular mask with 1-pixel antialiased margin
            mask_arr = np.full((height, width), 255, dtype=np.uint8)
            if width > 4 and height > 4:
                mask_arr[0, :] = 180
                mask_arr[-1, :] = 180
                mask_arr[:, 0] = 180
                mask_arr[:, -1] = 180
            return Image.fromarray(mask_arr, mode="L")

        y_coords, x_coords = np.ogrid[:height, :width]
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        rx = max(1.0, width / 2.0)
        ry = max(1.0, height / 2.0)

        # Normalized radial distance squared
        norm_dist = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2

        if mask_type == BrushMaskType.ELLIPSE:
            # Antialiased ellipse with smooth outer falloff
            mask_float = np.clip((1.05 - norm_dist) / 0.15, 0.0, 1.0) * 255.0
            return Image.fromarray(mask_float.astype(np.uint8), mode="L")

        # BrushMaskType.ORGANIC_STROKE:
        # Elliptical core modulated by subtle brush edge texture / falloff
        core = np.clip((1.1 - norm_dist) / 0.25, 0.0, 1.0)
        # Subtle horizontal brush striation modulation
        x_norm = np.abs((x_coords - cx) / rx)
        falloff = 1.0 - 0.2 * (x_norm ** 4)
        mask_float = np.clip(core * falloff * 255.0, 0.0, 255.0)
        return Image.fromarray(mask_float.astype(np.uint8), mode="L")

    def _apply_color_correction(
        self,
        src_rgb: Image.Image,
        target_lab: np.ndarray,
        mode: ColorCorrectionMode,
        strength: float,
        lum_preservation: float,
    ) -> Image.Image:
        """Adjust source colors towards the target region while preserving inner image texture."""
        if mode == ColorCorrectionMode.NONE or strength <= 0.001:
            return src_rgb

        arr_rgb = np.array(src_rgb, dtype=np.float32) / 255.0
        src_lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)

        src_mean_l = float(np.mean(src_lab[:, :, 0]))
        src_mean_a = float(np.mean(src_lab[:, :, 1]))
        src_mean_b = float(np.mean(src_lab[:, :, 2]))

        tgt_l, tgt_a, tgt_b = target_lab[0], target_lab[1], target_lab[2]

        if mode in (ColorCorrectionMode.MEAN_SHIFT, ColorCorrectionMode.REINHARD):
            # Effective target means scaled by color_strength
            eff_l = (1.0 - strength) * src_mean_l + strength * tgt_l
            eff_a = (1.0 - strength) * src_mean_a + strength * tgt_a
            eff_b = (1.0 - strength) * src_mean_b + strength * tgt_b

            out_lab = src_lab.copy()

            # For luminance: preserve high-frequency texture differences from source mean
            lum_diff = src_lab[:, :, 0] - src_mean_l
            # Scale internal contrast slightly to preserve visibility
            out_lab[:, :, 0] = eff_l + lum_diff * (0.6 + 0.4 * lum_preservation)

            # Chrominance shift
            out_lab[:, :, 1] = eff_a + (src_lab[:, :, 1] - src_mean_a)
            out_lab[:, :, 2] = eff_b + (src_lab[:, :, 2] - src_mean_b)

            # Clamp CIELAB ranges
            out_lab[:, :, 0] = np.clip(out_lab[:, :, 0], 0.0, 100.0)
            out_lab[:, :, 1] = np.clip(out_lab[:, :, 1], -127.0, 127.0)
            out_lab[:, :, 2] = np.clip(out_lab[:, :, 2], -127.0, 127.0)

            # Convert back to RGB float32
            out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
            out_rgb_uint8 = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(out_rgb_uint8, mode="RGB")

        if mode == ColorCorrectionMode.TINT:
            # Linear tint in RGB
            # Convert target LAB to RGB
            dummy_lab = np.array([[[tgt_l, tgt_a, tgt_b]]], dtype=np.float32)
            tgt_rgb = np.clip(cv2.cvtColor(dummy_lab, cv2.COLOR_LAB2RGB)[0, 0], 0.0, 1.0)
            tinted = (1.0 - strength) * arr_rgb + strength * tgt_rgb[None, None, :]
            tinted_uint8 = np.clip(tinted * 255.0, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(tinted_uint8, mode="RGB")

        return src_rgb

    def _apply_refinements(
        self,
        mosaic_rgb: Image.Image,
        analyzer: PaintingAnalyzer,
    ) -> Image.Image:
        """Apply multi-scale composition correction and optional blending with original painting."""
        if self.config.original_blend <= 0.001 and self.config.refinement_passes <= 0:
            return mosaic_rgb

        w, h = mosaic_rgb.size
        target_img = analyzer.target_image.resize((w, h), Image.Resampling.BILINEAR)

        result = mosaic_rgb

        # Phase E: Low-resolution multi-scale composition correction
        if self.config.refinement_passes > 0:
            # Extract low-frequency target illumination/color differences
            target_blur = target_img.filter(ImageFilter.GaussianBlur(radius=max(8, int(w / 32))))
            mosaic_blur = result.filter(ImageFilter.GaussianBlur(radius=max(8, int(w / 32))))

            t_blur_np = np.array(target_blur, dtype=np.float32)
            m_blur_np = np.array(mosaic_blur, dtype=np.float32)
            res_np = np.array(result, dtype=np.float32)

            # Multi-scale composition residual: inject gentle low-frequency adjustment
            # L_global correction factor
            correction_factor = 0.15
            refined_np = res_np + correction_factor * (t_blur_np - m_blur_np)
            refined_uint8 = np.clip(refined_np, 0.0, 255.0).astype(np.uint8)
            result = Image.fromarray(refined_uint8, mode="RGB")

        # Blend with original painting if configured
        if self.config.original_blend > 0.001:
            blend_alpha = float(np.clip(self.config.original_blend, 0.0, 1.0))
            result = Image.blend(result, target_img, alpha=blend_alpha)

        return result
