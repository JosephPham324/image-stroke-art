"""Tests for descriptor extraction and vectorized matching distance."""

from __future__ import annotations

import numpy as np
import pytest

from image_mosaic.config import MatchWeights
from image_mosaic.descriptors import (
    DescriptorExtractor,
    ImageDescriptor,
    compute_vectorized_match_scores,
)


def test_descriptor_extraction_dimensions():
    extractor = DescriptorExtractor()
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    desc = extractor.extract_from_rgb(rgb)

    assert desc.mean_lab.shape == (3,)
    assert desc.luminance_stats.shape == (2,)
    assert desc.color_hist.shape == (24,)
    assert isinstance(desc.edge_density, float)
    assert desc.gradient_hist.shape == (8,)

    vec = desc.to_vector()
    assert vec.shape == (38,)


def test_pure_white_and_black_luminance():
    extractor = DescriptorExtractor()

    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    desc_w = extractor.extract_from_rgb(white)
    assert pytest.approx(desc_w.mean_lab[0], rel=0.05) == 100.0

    black = np.zeros((16, 16, 3), dtype=np.uint8)
    desc_b = extractor.extract_from_rgb(black)
    assert pytest.approx(desc_b.mean_lab[0], abs=0.5) == 0.0


def test_vectorized_matching_selects_closest_color():
    extractor = DescriptorExtractor()

    # Red, Green, Blue source images
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    green = np.zeros((16, 16, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    blue = np.zeros((16, 16, 3), dtype=np.uint8)
    blue[:, :, 2] = 255

    desc_r = extractor.extract_from_rgb(red)
    desc_g = extractor.extract_from_rgb(green)
    desc_b = extractor.extract_from_rgb(blue)

    matrix = np.vstack([desc_r.to_vector(), desc_g.to_vector(), desc_b.to_vector()])

    # Query with reddish color
    query_img = np.zeros((16, 16, 3), dtype=np.uint8)
    query_img[:, :, 0] = 230
    query_img[:, :, 1] = 20
    query_desc = extractor.extract_from_rgb(query_img)

    weights = MatchWeights(color=1.0, luminance=0.0, histogram=0.0, texture=0.0, reuse=0.0)
    scores = compute_vectorized_match_scores(query_desc, matrix, weights)

    assert int(np.argmin(scores)) == 0  # Red is closest


def test_reuse_penalty_shifts_choice():
    extractor = DescriptorExtractor()
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    red2 = np.zeros((16, 16, 3), dtype=np.uint8)
    red2[:, :, 0] = 250
    red2[:, :, 1] = 10

    desc1 = extractor.extract_from_rgb(red)
    desc2 = extractor.extract_from_rgb(red2)
    matrix = np.vstack([desc1.to_vector(), desc2.to_vector()])

    query_desc = desc1
    weights = MatchWeights(color=1.0, luminance=0.0, histogram=0.0, texture=0.0, reuse=1.0)

    # First without reuse: index 0 wins
    scores_no_reuse = compute_vectorized_match_scores(query_desc, matrix, weights, reuse_counts=np.array([0, 0]))
    assert int(np.argmin(scores_no_reuse)) == 0

    # With high reuse on index 0: index 1 wins due to penalty
    scores_with_reuse = compute_vectorized_match_scores(query_desc, matrix, weights, reuse_counts=np.array([50, 0]))
    assert int(np.argmin(scores_with_reuse)) == 1
