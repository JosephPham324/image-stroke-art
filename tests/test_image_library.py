"""Tests for source image library loading, deduplication, and error handling."""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from image_mosaic.image_library import EmptyLibraryError, ImageLibrary


def test_empty_directory_raises_error(tmp_path: Path):
    empty_dir = tmp_path / "empty_sources"
    empty_dir.mkdir()
    lib = ImageLibrary(sources_dir=empty_dir)
    with pytest.raises(EmptyLibraryError):
        lib.load()


def test_nonexistent_directory_raises_error(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    lib = ImageLibrary(sources_dir=nonexistent)
    with pytest.raises(FileNotFoundError):
        lib.load()


def test_loads_valid_images_and_skips_corrupt_and_duplicates(synthetic_sources_dir: Path):
    lib = ImageLibrary(sources_dir=synthetic_sources_dir)
    lib.load()

    # 8 distinct color textures + 1 transparent face = 9 unique valid images
    # Corrupt file skipped, duplicate file skipped
    assert lib.count == 9
    assert lib.descriptors_matrix.shape == (9, 38)


def test_transparent_image_detected(synthetic_sources_dir: Path):
    lib = ImageLibrary(sources_dir=synthetic_sources_dir)
    lib.load()

    transparent_entries = [img for img in lib.images if img.has_alpha]
    assert len(transparent_entries) == 1
    assert "transparent_face" in transparent_entries[0].path.name


def test_get_source_pil_resizing(synthetic_sources_dir: Path):
    lib = ImageLibrary(sources_dir=synthetic_sources_dir)
    lib.load()

    img = lib.get_source_pil(0, target_size=(32, 24))
    assert img.size == (32, 24)
