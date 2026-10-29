"""Perceptual color and texture descriptor extraction and distance matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from image_mosaic.config import MatchWeights


@dataclass
class ImageDescriptor:
    """Rich descriptor of perceptual color, luminance distribution, and texture."""
    mean_lab: np.ndarray        # shape: (3,) float32 [L*, a*, b*]
    luminance_stats: np.ndarray # shape: (2,) float32 [mean_L, std_L]
    color_hist: np.ndarray      # shape: (24,) float32 (8 bins L, 8 bins a, 8 bins b)
    edge_density: float         # float32 in [0, 1]
    gradient_hist: np.ndarray   # shape: (8,) float32 orientation distribution [0, pi]

    def to_vector(self) -> np.ndarray:
        """Flatten into a single 1D feature array of length 38."""
        return np.concatenate([
            self.mean_lab,
            self.luminance_stats,
            self.color_hist,
            np.array([self.edge_density], dtype=np.float32),
            self.gradient_hist,
        ])


class DescriptorExtractor:
    """Extracts perceptual Lab color and texture features from RGB images or patches."""

    def __init__(
        self,
        num_hist_bins: int = 8,
        gradient_bins: int = 8,
        edge_threshold: float = 0.15,
    ) -> None:
        self.num_hist_bins = num_hist_bins
        self.gradient_bins = gradient_bins
        self.edge_threshold = edge_threshold

    def extract_from_rgb(self, rgb_image: np.ndarray) -> ImageDescriptor:
        """Extract all descriptor components from an RGB image array (H, W, 3).
        
        rgb_image can be uint8 [0, 255] or float32 [0.0, 1.0].
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {rgb_image.shape}")

        if rgb_image.dtype == np.uint8:
            rgb_float = (rgb_image.astype(np.float32) / 255.0)
        else:
            rgb_float = np.clip(rgb_image.astype(np.float32), 0.0, 1.0)

        # Convert to CIELAB space
        lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]  # [0, 100]
        a_chan = lab[:, :, 1]  # [-128, 127]
        b_chan = lab[:, :, 2]  # [-128, 127]

        # 1. Mean perceptual color
        mean_l = float(np.mean(l_chan))
        mean_a = float(np.mean(a_chan))
        mean_b = float(np.mean(b_chan))
        mean_lab = np.array([mean_l, mean_a, mean_b], dtype=np.float32)

        # 2. Luminance statistics
        std_l = float(np.std(l_chan))
        luminance_stats = np.array([mean_l, std_l], dtype=np.float32)

        # 3. Marginal color histograms (8 bins per channel)
        hist_l, _ = np.histogram(l_chan, bins=self.num_hist_bins, range=(0.0, 100.0))
        hist_a, _ = np.histogram(a_chan, bins=self.num_hist_bins, range=(-100.0, 100.0))
        hist_b, _ = np.histogram(b_chan, bins=self.num_hist_bins, range=(-100.0, 100.0))

        # Normalize each marginal histogram to sum to 1.0
        sum_l = hist_l.sum()
        norm_l = hist_l.astype(np.float32) / (sum_l if sum_l > 0 else 1.0)
        sum_a = hist_a.sum()
        norm_a = hist_a.astype(np.float32) / (sum_a if sum_a > 0 else 1.0)
        sum_b = hist_b.sum()
        norm_b = hist_b.astype(np.float32) / (sum_b if sum_b > 0 else 1.0)
        color_hist = np.concatenate([norm_l, norm_a, norm_b])

        # 4. Texture & Edge density using luminance
        gray = l_chan / 100.0  # normalize luminance to [0, 1]
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Edge density: proportion of pixels above edge threshold
        edge_density = float(np.mean(grad_mag > self.edge_threshold))

        # Gradient orientation histogram weighted by magnitude: [0, pi]
        grad_angle = np.mod(np.arctan2(grad_y, grad_x), np.pi)
        grad_hist, _ = np.histogram(
            grad_angle,
            bins=self.gradient_bins,
            range=(0.0, float(np.pi)),
            weights=grad_mag,
        )
        grad_sum = grad_hist.sum()
        norm_grad_hist = grad_hist.astype(np.float32) / (grad_sum if grad_sum > 0 else 1.0)

        return ImageDescriptor(
            mean_lab=mean_lab,
            luminance_stats=luminance_stats,
            color_hist=color_hist,
            edge_density=edge_density,
            gradient_hist=norm_grad_hist,
        )

    def extract_patch(
        self,
        lab_image: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> ImageDescriptor:
        """Fast patch extraction from precomputed Lab image."""
        h, w = lab_image.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        patch_lab = lab_image[y0:y1, x0:x1]
        l_chan = patch_lab[:, :, 0]
        a_chan = patch_lab[:, :, 1]
        b_chan = patch_lab[:, :, 2]

        mean_l = float(np.mean(l_chan))
        mean_a = float(np.mean(a_chan))
        mean_b = float(np.mean(b_chan))
        mean_lab = np.array([mean_l, mean_a, mean_b], dtype=np.float32)

        std_l = float(np.std(l_chan))
        luminance_stats = np.array([mean_l, std_l], dtype=np.float32)

        hist_l, _ = np.histogram(l_chan, bins=self.num_hist_bins, range=(0.0, 100.0))
        hist_a, _ = np.histogram(a_chan, bins=self.num_hist_bins, range=(-100.0, 100.0))
        hist_b, _ = np.histogram(b_chan, bins=self.num_hist_bins, range=(-100.0, 100.0))

        sum_l = hist_l.sum()
        norm_l = hist_l.astype(np.float32) / (sum_l if sum_l > 0 else 1.0)
        sum_a = hist_a.sum()
        norm_a = hist_a.astype(np.float32) / (sum_a if sum_a > 0 else 1.0)
        sum_b = hist_b.sum()
        norm_b = hist_b.astype(np.float32) / (sum_b if sum_b > 0 else 1.0)
        color_hist = np.concatenate([norm_l, norm_a, norm_b])

        gray = l_chan / 100.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        edge_density = float(np.mean(grad_mag > self.edge_threshold))

        grad_angle = np.mod(np.arctan2(grad_y, grad_x), np.pi)
        grad_hist, _ = np.histogram(
            grad_angle,
            bins=self.gradient_bins,
            range=(0.0, float(np.pi)),
            weights=grad_mag,
        )
        grad_sum = grad_hist.sum()
        norm_grad_hist = grad_hist.astype(np.float32) / (grad_sum if grad_sum > 0 else 1.0)

        return ImageDescriptor(
            mean_lab=mean_lab,
            luminance_stats=luminance_stats,
            color_hist=color_hist,
            edge_density=edge_density,
            gradient_hist=norm_grad_hist,
        )


class MatchScoreWeights(MatchWeights):
    """Backward compatibility alias for MatchWeights."""
    pass


def compute_vectorized_match_scores(
    target_desc: ImageDescriptor,
    source_matrix: np.ndarray,
    weights: MatchWeights,
    reuse_counts: Optional[np.ndarray] = None,
    max_reuse: int = 0,
) -> np.ndarray:
    """Compute weighted distance scores between a target descriptor and N source descriptors.
    
    Vectorized across all N sources using NumPy.
    
    source_matrix has shape (N, 38):
      - 0:3   : mean_lab (3)
      - 3:5   : luminance_stats (2)
      - 5:29  : color_hist (24)
      - 29:30 : edge_density (1)
      - 30:38 : gradient_hist (8)
      
    Returns:
      scores: shape (N,) float32. Lower is better.
    """
    n_sources = source_matrix.shape[0]
    if n_sources == 0:
        return np.empty((0,), dtype=np.float32)

    scores = np.zeros(n_sources, dtype=np.float32)

    # 1. Perceptual color distance (Delta E Euclidean in CIELAB, normalized by 100)
    if weights.color > 0:
        src_lab = source_matrix[:, 0:3]
        d_color = np.linalg.norm(src_lab - target_desc.mean_lab[None, :], axis=1) / 100.0
        scores += weights.color * d_color

    # 2. Luminance distance (mean + std mismatch, normalized)
    if weights.luminance > 0:
        src_lum = source_matrix[:, 3:5]
        diff_mean_l = np.abs(src_lum[:, 0] - target_desc.luminance_stats[0]) / 100.0
        diff_std_l = np.abs(src_lum[:, 1] - target_desc.luminance_stats[1]) / 50.0
        scores += weights.luminance * (diff_mean_l + 0.5 * diff_std_l)

    # 3. Histogram distance (0.5 * L1 distance across 3 marginals, normalized to [0, 1])
    if weights.histogram > 0:
        src_hist = source_matrix[:, 5:29]
        d_hist = 0.5 * np.sum(np.abs(src_hist - target_desc.color_hist[None, :]), axis=1) / 3.0
        scores += weights.histogram * d_hist

    # 4. Texture distance (edge density difference + gradient orientation L1)
    if weights.texture > 0:
        src_edge = source_matrix[:, 29]
        src_grad = source_matrix[:, 30:38]
        d_edge = np.abs(src_edge - target_desc.edge_density)
        d_grad = 0.5 * np.sum(np.abs(src_grad - target_desc.gradient_hist[None, :]), axis=1)
        scores += weights.texture * (0.6 * d_edge + 0.4 * d_grad)

    # 5. Reuse penalty
    if reuse_counts is not None and weights.reuse > 0:
        # Scale penalty smoothly with usage
        penalty = reuse_counts.astype(np.float32) / (np.max(reuse_counts) + 1.0)
        scores += weights.reuse * penalty

        # If a hard ceiling is specified, heavily penalize exhausted images
        if max_reuse > 0:
            exhausted = reuse_counts >= max_reuse
            scores[exhausted] += 1e6

    return scores
