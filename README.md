# Composition-Aware Image Brush Mosaic Generator

A high-performance Python computer vision application that reconstructs a target painting or photograph using a collection of source images as textured, directional visual brushstrokes.

When viewed from a distance, the composite faithfully retains the target image's broad composition, lighting, silhouettes, and color distribution. When inspected up close, the individual source images are revealed as the building blocks and visual brushstrokes of the artwork.

---

## Table of Contents

1. [Key Features & Concept](#key-features--concept)
2. [Uniform-Grid Photomosaic vs. Directional Brush Mosaic](#uniform-grid-photomosaic-vs-directional-brush-mosaic)
3. [Algorithmic Architecture](#algorithmic-architecture)
   - [Target Painting Analysis](#1-target-painting-analysis)
   - [Rich Perceptual Descriptors & Matching](#2-rich-perceptual-descriptors--matching)
   - [Spatially Adaptive Multi-Scale Stroke Planning](#3-spatially-adaptive-multi-scale-stroke-planning)
   - [Rendering, Masking & Color Correction](#4-rendering-masking--color-correction)
   - [Composition Preservation & Refinement Pass](#5-composition-preservation--refinement-pass)
4. [Installation](#installation)
5. [Quickstart & Example Usage](#quickstart--example-usage)
6. [Command-Line Interface (CLI) Reference](#command-line-interface-cli-reference)
7. [Configuration Reference](#configuration-reference)
8. [Performance & Scalability](#performance--scalability)
9. [Automated Testing](#automated-testing)
10. [Troubleshooting & Limitations](#troubleshooting--limitations)

---

## Key Features & Concept

- **Dual Rendering Modes**:
  - **Baseline Grid Mode (Phase A)**: Regular uniform-grid photomosaic for fast baseline rendering.
  - **Adaptive Brush Mode (Phases C-F)**: Spatially adaptive, multi-scale, directional brushstrokes.
- **Saliency-Guided Scale Adaptation**: Places broad, sweeping strokes across uniform backgrounds (sky, calm waters, flat walls) and high-density, miniature strokes along sharp edges, facial landmarks, and intricate textures.
- **Vectorized Gradient & Tangent Orientation Field**: Derives local contour tangents from 3x3 Scharr gradients, smoothly flowing brushstrokes along natural artistic boundaries.
- **CIELAB Perceptual Color Matching**: Evaluates perceptual color differences in CIE $L^*a^*b^*$ space rather than naive Euclidean RGB.
- **Internal Texture & Luminance Preservation**: Adjusts the mean chrominance of the source image to match the target while preserving the source image's internal contrast, recognizable features, and textures.
- **Repetition Control & Entropy Balancing**: Vectorized reuse penalties prevent visual monotony and allow configurable limits on source image repetition.
- **Masking & Organic Feathering**: Supports rectangular, elliptical, and organic feathered brush masks with antialiased edge falloff.
- **Robust Source Image Ingestion**: Recursively discovers images, respects EXIF orientation tags, filters duplicates via SHA-256 content hashing, and handles transparent PNGs (RGBA) gracefully.

---

## Uniform-Grid Photomosaic vs. Directional Brush Mosaic

Traditional photomosaics divide a target into a rigid grid of identical square tiles and select source images based solely on mean RGB Euclidean distance:

$$\text{Distance}_{\text{RGB}} = \sqrt{(R_t - R_s)^2 + (G_t - G_s)^2 + (B_t - B_s)^2}$$

### Limitations of Average Color Alone

1. **Loss of Structure and Edges**: A tile with half black and half white has the same average color as a solid 50% gray tile, yet placed over an edge, a solid gray tile obliterates the boundary.
2. **Perceptual Non-Uniformity**: The human visual system does not perceive color differences uniformly in RGB space (e.g. green variations appear much brighter than blue variations).
3. **Rigid Grid Artifacts**: Distinct vertical and horizontal seams draw the eye away from the artwork's natural organic contours.

### The Directional Brush Approach

The Composition-Aware Image Brush Mosaic generator addresses these limitations by:

1. **Multi-Scale Spatial Allocation**: Using smaller strokes in high-frequency regions (silhouettes, eyes, contours) and larger strokes in smooth gradients.
2. **Directional Flow Along Tangents**: Aligning strokes with the tangent of local image gradients ($\theta_{\text{tangent}} = \theta_{\text{gradient}} + \pi/2$), mimicking how artists sweep brush bristles along forms.
3. **Multi-Attribute Descriptors**: Combining CIELAB color, luminance standard deviation, marginal color histograms, edge density, and gradient orientation histograms into a normalized matching score.
4. **Feathered Organic Overlap**: Layering strokes with controlled overlap and feathered boundaries to eliminate tile seams and prevent visible canvas gaps.

---

## Algorithmic Architecture

```
Target Image                     Source Images Directory
     │                                     │
     ▼                                     ▼
PaintingAnalyzer                      ImageLibrary
- CIELAB conversion                   - Recursive search & EXIF fix
- Scharr gradients & tangents         - SHA-256 deduplication
- Multiscale saliency map             - Thumbnail cache & descriptors
     │                                     │
     └───────────────────┬─────────────────┘
                         │
                         ▼
                   StrokePlanner
          - Grid / Adaptive Multi-Scale
          - Vectorized candidate matching
          - Saliency-based sizing
          - Contour tangent orientation
          - Reuse penalty tracking
                         │
                         ▼
                    StrokePlan
                         │
                         ▼
                  MosaicRenderer
          - Resize & rotate source image
          - Organic / Elliptical masking
          - Reinhard / Lab mean shift
          - Alpha compositing
          - Composition refinement pass
                         │
                         ▼
                    Final Image
```

### 1. Target Painting Analysis
The `PaintingAnalyzer` computes:
- Standardized float32 CIELAB color representation ($L^* \in [0, 100], a^* \in [-128, 127], b^* \in [-128, 127]$).
- Normalized luminance channel $L^*/100.0$.
- Scharr first-order horizontal and vertical gradients ($G_x, G_y$), gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$, and contour tangent angle $\theta_t = \text{arctan2}(G_y, G_x) + \pi/2$.
- Perceptual saliency map: Combines normalized edge magnitude with local luminance variance $\text{Var}(I) = \mathbb{E}[I^2] - (\mathbb{E}[I])^2$, followed by a Gaussian smoothing filter.

### 2. Rich Perceptual Descriptors & Matching
Each image and target patch is characterized by a 38-dimensional feature vector:
1. **Mean CIELAB Color** (3 floats): $L^*, a^*, b^*$.
2. **Luminance Statistics** (2 floats): $\mu_L, \sigma_L$.
3. **Color Distribution** (24 floats): 8-bin marginal histograms for $L^*$, $a^*$, and $b^*$.
4. **Edge Density** (1 float): Proportion of pixels with gradient magnitude exceeding threshold.
5. **Gradient Orientation Histogram** (8 floats): Magnitude-weighted orientation bins over $[0, \pi]$.

Matching is evaluated across all $N$ source candidates simultaneously using vectorized NumPy broadcasting:

$$\text{Score} = w_{\text{color}} d_{\text{color}} + w_{\text{luminance}} d_{\text{luminance}} + w_{\text{histogram}} d_{\text{histogram}} + w_{\text{texture}} d_{\text{texture}} + w_{\text{reuse}} p_{\text{reuse}}$$

Where all distance terms are normalized to approximately $[0, 1]$ before applying user-configured weights.

### 3. Spatially Adaptive Multi-Scale Stroke Planning
The `AdaptiveStrokePlanner` structures stroke layout into coarse-to-fine layers:
- **Layer 0 (Underpainting / Background)**: Large strokes ($\approx \text{max\_stroke\_size}$) covering low-saliency, broad regions to guarantee 100% canvas coverage without gaps.
- **Layer 1 (Mid-scale Features)**: Intermediate strokes capturing medium-frequency color transitions.
- **Layer 2 (Detail & Edges)**: Fine strokes ($\approx \text{min\_stroke\_size}$) concentrated around sharp contours, facial features, and high saliency.

Strokes are oriented along local tangent angles when edge magnitude is significant, and deterministic pseudo-random jitter is introduced for natural brush texture.

### 4. Rendering, Masking & Color Correction
The `MosaicRenderer` executes the planned strokes:
- **Brush Masks**: Generates antialiased 8-bit alpha masks (`rectangle`, `ellipse`, `organic_stroke`).
- **Color Correction**: Adjusts mean color while preserving inner texture:
  - **Reinhard Transfer / Lab Mean Shift**: Moves the mean $L^*, a^*, b^*$ to match the target while preserving the internal deviation $(Lab_s - \mu_s)$. Luminance preservation preserves internal texture and prevents over-saturation.
  - **Safe Clamping**: Ensures values are clamped to $[0, 255]$ without color clipping artifacts or channel overflow.
- **Compositing**: Rotates the masked RGBA stroke tile with bilinear interpolation and composites it onto the canvas using Porter-Duff Over operations.

### 5. Composition Preservation & Refinement Pass
After all strokes are placed:
- **Multi-Scale Composition Correction**: Measures the low-frequency illumination/color residual between a Gaussian-blurred target and the mosaic, gently injecting missing low-frequency ambient tone.
- **Original Image Blend**: Optional linear blending with the original target image ($\alpha \in [0.0, 1.0]$) to fine-tune the balance between recognizable portraiture and visible source mosaic tiles.

---

## Installation

### Prerequisites
- Python 3.11 or newer (Python 3.12 supported).
- Windows, macOS, or Linux.

### Install Dependencies
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/image-stroke-art.git
cd image-stroke-art

# Install using pip
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

Core dependencies: `Pillow`, `numpy`, `opencv-python`, `scikit-image`, `scipy`, `typer`, `rich`, `tqdm`, `pytest`.

---

## Quickstart & Example Usage

Run the built-in generator script to synthesize sample artwork and compare baseline vs. adaptive rendering:

```bash
py examples/generate_samples.py
```

### Basic CLI Command (Adaptive Mode)

```bash
py -m image_mosaic \
    --target path/to/painting.jpg \
    --sources path/to/source_folder \
    --output output_mosaic.png \
    --mode adaptive \
    --min-stroke-size 12 \
    --max-stroke-size 44 \
    --color-strength 0.85 \
    --original-blend 0.15 \
    --seed 42
```

### Baseline Grid Command

```bash
py -m image_mosaic \
    --target path/to/painting.jpg \
    --sources path/to/source_folder \
    --output output_grid.png \
    --mode grid \
    --tile-size 24 \
    --seed 42
```

---

## Command-Line Interface (CLI) Reference

```text
Usage: python -m image_mosaic [OPTIONS]

Options:
  -t, --target PATH              Path to the target painting or image to reconstruct. [required]
  -s, --sources PATH             Directory containing source images to use as brushstrokes. [required]
  -o, --output PATH              Path where the final mosaic image will be saved. [default: mosaic_output.png]
  -p, --preview PATH             Optional path to save a downscaled preview image.
  -m, --mode [grid|adaptive]     Rendering mode. [default: adaptive]
      --tile-size INT            Tile dimension in pixels for grid mode. [default: 24]
      --min-stroke-size INT      Minimum brushstroke dimension in pixels for detail regions. [default: 12]
      --max-stroke-size INT      Maximum brushstroke dimension in pixels for background regions. [default: 48]
      --stroke-aspect-ratio FLOAT Aspect ratio (width / height) for elongated brushstrokes. [default: 1.2]
      --brush-mask [rectangle|ellipse|organic_stroke] Stroke mask contour. [default: organic_stroke]
      --color-correction [reinhard|mean_shift|tint|none] Color adaptation mode. [default: reinhard]
      --color-strength FLOAT     Strength of color correction (0.0 to 1.0). [default: 0.85]
      --luminance-preservation FLOAT Preservation of internal contrast (0.0 to 1.0). [default: 0.5]
      --original-blend FLOAT     Blend strength of original painting (0.0 to 1.0). [default: 0.15]
      --directional / --no-directional Align strokes along local edge contours. [default: directional]
      --max-rotation FLOAT       Maximum stroke rotation in degrees. [default: 90.0]
      --overlap-ratio FLOAT      Overlap ratio between adjacent brushstrokes. [default: 0.25]
      --max-reuse INT            Maximum allowed usage count per source image (0 = unlimited). [default: 0]
      --reuse-penalty FLOAT      Penalty factor discouraging repeated source images. [default: 0.3]
      --w-color FLOAT            Matching weight for mean Lab color. [default: 1.0]
      --w-luminance FLOAT        Matching weight for luminance stats. [default: 0.5]
      --w-hist FLOAT             Matching weight for color histogram. [default: 0.3]
      --w-texture FLOAT          Matching weight for texture and gradient. [default: 0.2]
      --seed INT                 Random seed for reproducible layout. [default: 42]
      --max-resolution INT       Optionally downscale target image if max dimension exceeds this value.
      --refinement-passes INT    Number of multi-scale composition refinement passes. [default: 1]
  -v, --verbose                  Enable verbose debug logging.
      --help                     Show this message and exit.
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `RenderingMode` | `adaptive` | `grid` (uniform squares) or `adaptive` (multi-scale brushstrokes) |
| `tile_size` | `int` | `24` | Tile width/height in pixels for grid mode |
| `min_stroke_size` | `int` | `12` | Stroke size in high-detail / edge regions |
| `max_stroke_size` | `int` | `48` | Stroke size in broad / background regions |
| `stroke_aspect_ratio`| `float` | `1.2` | Elongation ratio (width / height) along brush direction |
| `brush_mask` | `BrushMaskType` | `organic_stroke` | Mask shape (`rectangle`, `ellipse`, `organic_stroke`) |
| `color_correction` | `ColorCorrectionMode`| `reinhard` | `reinhard`, `mean_shift`, `tint`, or `none` |
| `color_strength` | `float` | `0.85` | Degree of color adaptation towards target (0.0 = none, 1.0 = full) |
| `luminance_preservation` | `float` | `0.5` | Contrast preservation for internal source textures |
| `original_blend` | `float` | `0.15` | Final blend factor with original painting |
| `enable_directional` | `bool` | `True` | Aligns strokes along image gradient tangents |
| `reuse_penalty_weight` | `float` | `0.3` | Weight discouraging excessive source repetition |
| `max_reuse_count` | `int` | `0` | Hard cap on usages per source image (0 = unlimited) |
| `seed` | `int` | `42` | Random seed for exact reproducibility |

---

## Performance & Scalability

- **Precomputed Descriptor Cache**: All source images are indexed and converted into a compact NumPy matrix $(N, 38)$ once at startup.
- **Vectorized NumPy Matching**: Candidate matching uses array broadcasting; comparing a target patch against 1,000 source images executes in sub-millisecond time.
- **Memory Footprint Optimization**: Source images are downscaled into standard thumbnail representations for indexing and memory caching. Full-resolution originals are loaded on-demand only if required.
- **Progress Reporting & Non-blocking Diagnostics**: Real-time console status spinners and execution timings are displayed for all major processing stages.

---

## Automated Testing

The project includes a comprehensive suite of unit, integration, and end-to-end tests using synthetic fixtures:

```bash
# Run the test suite
py -m pytest -v
```

### Test Coverage Areas

- `tests/test_image_library.py`: Image loading, EXIF rotation, RGBA transparency, corrupted file handling, duplicate suppression, empty directory validation.
- `tests/test_descriptors.py`: CIELAB conversion, luminance statistics, color histogram normalization, edge density, vectorized distance scoring, reuse penalty behavior.
- `tests/test_stroke_planner.py`: Grid tile planning, multi-scale adaptive planning, layer hierarchy, deterministic seed reproducibility.
- `tests/test_renderer.py`: Mask generation, Reinhard color adaptation, luminance preservation, boundary clipping safety.
- `tests/test_cli.py`: Typer argument validation, missing option errors, invalid directory reporting, full CLI execution.
- `tests/test_end_to_end.py`: Synthetic end-to-end generation, dimension verification, preview generation, and diagnostic metric assertions (PSNR, CIELAB Mean $\Delta E$, repetition entropy).

---

## Troubleshooting & Limitations

### Common Issues

1. **"No supported images found in source directory"**:
   - Ensure the `--sources` folder contains supported formats (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`).
   - If images are in subdirectories, they are scanned recursively.

2. **Source Images Hard to Recognize**:
   - Reduce `--color-strength` (e.g. from `0.85` to `0.70`).
   - Increase `--luminance-preservation` (e.g. from `0.5` to `0.75`).
   - Increase `--min-stroke-size` and `--max-stroke-size`.

3. **Target Painting Composition Looks Blurry**:
   - Decrease `--min-stroke-size` (e.g. to `8` or `10`).
   - Increase `--color-strength` (e.g. `0.90`).
   - Enable `--directional` to ensure edges are tracked cleanly.
   - Adjust `--original-blend` (e.g. `0.15` to `0.20`).

4. **Too Many Repetitions of the Same Image**:
   - Increase `--reuse-penalty` (e.g. from `0.3` to `0.8` or `1.0`).
   - Set a hard cap with `--max-reuse <N>`.
