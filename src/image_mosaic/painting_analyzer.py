"""Analysis of target painting: multi-scale representation, gradients, tangents, and saliency."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from image_mosaic.descriptors import DescriptorExtractor, ImageDescriptor


class PaintingAnalyzer:
    """Extracts structural maps, gradient fields, and saliency from a target painting."""

    def __init__(
        self,
        image_or_path: Image.Image | str | Path,
        max_resolution: Optional[int] = None,
        extractor: Optional[DescriptorExtractor] = None,
    ) -> None:
        self.extractor = extractor or DescriptorExtractor()

        if isinstance(image_or_path, (str, bytes, Path)):
            with Image.open(image_or_path) as img:
                img = ImageOps.exif_transpose(img)
                self.original_image = img.convert("RGB")
        elif isinstance(image_or_path, Image.Image):
            self.original_image = ImageOps.exif_transpose(image_or_path).convert("RGB")
        else:
            raise TypeError(f"Expected Image, str, or Path, got {type(image_or_path)}")

        orig_w, orig_h = self.original_image.size

        # Optionally scale down very large images to max_resolution
        if max_resolution is not None and max(orig_w, orig_h) > max_resolution:
            scale = max_resolution / max(orig_w, orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            self.target_image = self.original_image.resize(
                (new_w, new_h), Image.Resampling.LANCZOS
            )
        else:
            self.target_image = self.original_image.copy()

        self.width, self.height = self.target_image.size

        # Convert to numpy float32 RGB [0, 1]
        self.rgb = np.array(self.target_image, dtype=np.float32) / 255.0

        # Convert to CIELAB float32
        self.lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2LAB)
        self.luminance = self.lab[:, :, 0]  # [0, 100]

        # Compute gradient vectors & magnitude
        self._compute_gradients()

        # Compute perceptual saliency map
        self._compute_saliency()

        # Compute low-resolution global composition map
        self._compute_global_composition()

    def _compute_gradients(self) -> None:
        """Calculate image gradients, edge magnitude, and stroke tangent angles."""
        gray = self.luminance / 100.0  # normalized to [0, 1]

        # Use 3x3 Scharr filter for rotation-invariant gradients
        self.grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        self.grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

        self.grad_magnitude = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2)
        max_mag = np.max(self.grad_magnitude)
        self.norm_grad_magnitude = self.grad_magnitude / (max_mag if max_mag > 1e-6 else 1.0)

        # Gradient orientation angle in radians [-pi, pi]
        self.grad_angle = np.arctan2(self.grad_y, self.grad_x)

        # Tangent angle (perpendicular to gradient, aligned along contours)
        # Brush strokes flow naturally along tangents
        self.tangent_angle = self.grad_angle + (np.pi / 2.0)
        # Wrap to [-pi, pi]
        self.tangent_angle = (self.tangent_angle + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_saliency(self) -> None:
        """Compute spatial saliency / detail density map.
        
        High values (near 1.0) indicate sharp edges, facial landmarks, or fine textures.
        Low values (near 0.0) indicate smooth, low-contrast backgrounds.
        """
        gray = self.luminance / 100.0

        # Local variance map: Var(I) = E[I^2] - (E[I])^2
        mean_filter = cv2.blur(gray, (7, 7))
        sq_filter = cv2.blur(gray ** 2, (7, 7))
        local_var = np.maximum(0.0, sq_filter - mean_filter ** 2)
        local_std = np.sqrt(local_var)
        max_std = np.max(local_std)
        norm_std = local_std / (max_std if max_std > 1e-6 else 1.0)

        # Combine edge magnitude with local contrast/variance
        raw_saliency = 0.65 * self.norm_grad_magnitude + 0.35 * norm_std

        # Apply mild Gaussian smoothing to avoid noisy single-pixel discontinuities
        smoothed = cv2.GaussianBlur(raw_saliency, (5, 5), sigmaX=1.2)
        self.saliency_map = np.clip(smoothed, 0.0, 1.0)

    def _compute_global_composition(self) -> None:
        """Compute downsampled multi-scale target representations."""
        # Scale to 64x64 or smaller for fast global composition checking
        low_res_size = (min(64, self.width), min(64, self.height))
        self.low_res_lab = cv2.resize(
            self.lab, low_res_size, interpolation=cv2.INTER_AREA
        )

    def get_patch_descriptor(self, x0: int, y0: int, x1: int, y1: int) -> ImageDescriptor:
        """Extract a descriptor for a target region."""
        return self.extractor.extract_patch(self.lab, x0, y0, x1, y1)

    def get_saliency_at(self, x: float, y: float, radius: float = 8.0) -> float:
        """Query peak and regional saliency around a given point."""
        ix = int(round(x))
        iy = int(round(y))
        r = max(1, int(round(radius)))

        x0 = max(0, ix - r)
        x1 = min(self.width, ix + r + 1)
        y0 = max(0, iy - r)
        y1 = min(self.height, iy + r + 1)

        patch = self.saliency_map[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.5
        return float(0.7 * np.max(patch) + 0.3 * np.mean(patch))

    def get_tangent_angle_degrees(self, x: float, y: float, radius: float = 6.0) -> Tuple[float, float]:
        """Get dominant stroke orientation (in degrees) and gradient strength at (x, y).
        
        Returns:
            (angle_deg, strength): angle in [-90, 90], strength in [0, 1].
        """
        ix = int(round(x))
        iy = int(round(y))
        r = max(1, int(round(radius)))

        x0 = max(0, ix - r)
        x1 = min(self.width, ix + r + 1)
        y0 = max(0, iy - r)
        y1 = min(self.height, iy + r + 1)

        sub_x = self.grad_x[y0:y1, x0:x1]
        sub_y = self.grad_y[y0:y1, x0:x1]
        sub_mag = self.norm_grad_magnitude[y0:y1, x0:x1]

        if sub_x.size == 0:
            return 0.0, 0.0

        mean_mag = float(np.mean(sub_mag))
        if mean_mag < 0.04:
            # Low gradient area: no strong edge orientation
            return 0.0, mean_mag

        # Weighted vector sum of tangents
        # Since orientations are pi-periodic (a stroke at theta is identical to theta + pi),
        # we double the angles to avoid cancellation in vector averaging:
        gx = np.mean(sub_x)
        gy = np.mean(sub_y)
        grad_theta = np.arctan2(gy, gx)
        tangent_theta = grad_theta + (np.pi / 2.0)

        # Convert to degrees in [-90, 90]
        deg = float(np.degrees(tangent_theta)) % 180.0
        if deg > 90.0:
            deg -= 180.0

        return deg, mean_mag
