"""Source image library loader, deduplication, thumbnailing, and descriptor cache."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from image_mosaic.descriptors import DescriptorExtractor, ImageDescriptor

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class EmptyLibraryError(ValueError):
    """Raised when no valid source images are found in the specified directory."""
    pass


@dataclass
class SourceImage:
    """Metadata and cached features for a single source image."""
    index: int
    path: Path
    content_hash: str
    width: int
    height: int
    aspect_ratio: float
    has_alpha: bool
    descriptor: ImageDescriptor
    thumbnail_rgb: np.ndarray  # (H, W, 3) uint8 RGB thumbnail for fast access


class ImageLibrary:
    """Manages the discovery, validation, caching, and querying of source images."""

    def __init__(
        self,
        sources_dir: Path,
        thumbnail_size: int = 128,
        extractor: Optional[DescriptorExtractor] = None,
    ) -> None:
        self.sources_dir = Path(sources_dir)
        self.thumbnail_size = thumbnail_size
        self.extractor = extractor or DescriptorExtractor()

        self.images: List[SourceImage] = []
        self._descriptors_matrix: Optional[np.ndarray] = None
        self._path_to_index: Dict[Path, int] = {}
        self._cache_thumbnails: Dict[int, Image.Image] = {}

    @property
    def count(self) -> int:
        return len(self.images)

    @property
    def descriptors_matrix(self) -> np.ndarray:
        if self._descriptors_matrix is None:
            if not self.images:
                self._descriptors_matrix = np.empty((0, 38), dtype=np.float32)
            else:
                vectors = [img.descriptor.to_vector() for img in self.images]
                self._descriptors_matrix = np.vstack(vectors).astype(np.float32)
        return self._descriptors_matrix

    def load(self, recursive: bool = True) -> None:
        """Scan sources_dir, validate, deduplicate, and compute descriptors."""
        if not self.sources_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.sources_dir}")
        if not self.sources_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.sources_dir}")

        pattern = "**/*" if recursive else "*"
        all_candidate_files = [
            p for p in self.sources_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not all_candidate_files:
            raise EmptyLibraryError(
                f"No supported images found in '{self.sources_dir}' (searched extensions: {SUPPORTED_EXTENSIONS})"
            )

        seen_hashes: Set[str] = set()
        self.images = []
        self._descriptors_matrix = None
        self._cache_thumbnails.clear()

        for path in sorted(all_candidate_files):
            try:
                # Fast duplicate check via MD5
                content_hash = self._compute_file_hash(path)
                if content_hash in seen_hashes:
                    logger.debug("Skipping duplicate file: %s", path)
                    continue

                # Load image safely
                with Image.open(path) as img:
                    # Apply EXIF rotation if present
                    img = ImageOps.exif_transpose(img)
                    orig_w, orig_h = img.size
                    if orig_w == 0 or orig_h == 0:
                        logger.warning("Skipping zero-sized image: %s", path)
                        continue

                    has_alpha = img.mode in ("RGBA", "LA") or (
                        img.mode == "P" and "transparency" in img.info
                    )

                    # Create RGB representation for descriptor computation
                    if has_alpha:
                        rgba = img.convert("RGBA")
                        # Composite onto neutral 50% gray background for fair color evaluation
                        bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
                        rgb_repr = Image.alpha_composite(bg, rgba).convert("RGB")
                    else:
                        rgb_repr = img.convert("RGB")

                    # Resize to standard thumbnail for memory efficiency & fast descriptor extraction
                    thumb = rgb_repr.resize(
                        (self.thumbnail_size, self.thumbnail_size), Image.Resampling.BILINEAR
                    )
                    thumb_np = np.array(thumb, dtype=np.uint8)

                    # Compute perceptual descriptor
                    descriptor = self.extractor.extract_from_rgb(thumb_np)

                    source_entry = SourceImage(
                        index=len(self.images),
                        path=path,
                        content_hash=content_hash,
                        width=orig_w,
                        height=orig_h,
                        aspect_ratio=float(orig_w) / float(orig_h),
                        has_alpha=has_alpha,
                        descriptor=descriptor,
                        thumbnail_rgb=thumb_np,
                    )

                    self.images.append(source_entry)
                    seen_hashes.add(content_hash)
                    self._path_to_index[path] = source_entry.index

            except Exception as e:
                logger.warning("Failed to load image '%s': %s", path, e)
                continue

        if not self.images:
            raise EmptyLibraryError(
                f"Found candidate files in '{self.sources_dir}', but none could be successfully loaded."
            )

        # Trigger precomputation of descriptors matrix
        _ = self.descriptors_matrix

    def get_source_pil(self, index: int, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Retrieve a source image as PIL Image.
        
        Loads on demand and resizes to target_size if requested.
        Preserves alpha channel if present.
        """
        if index < 0 or index >= len(self.images):
            raise IndexError(f"Source index {index} out of range (total: {len(self.images)})")

        entry = self.images[index]
        with Image.open(entry.path) as img:
            img = ImageOps.exif_transpose(img)
            if entry.has_alpha:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.BILINEAR)
            else:
                # Make a standalone copy so context manager closure does not invalidate
                img = img.copy()

        return img

    @staticmethod
    def _compute_file_hash(path: Path, block_size: int = 65536) -> str:
        """Compute SHA256 of file content to detect duplicates."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(block_size):
                hasher.update(chunk)
        return hasher.hexdigest()
