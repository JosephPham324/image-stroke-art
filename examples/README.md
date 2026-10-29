# Image Brush Mosaic Examples

This directory provides scripts and fixtures demonstrating both uniform-grid photomosaics (Phase A baseline) and composition-aware, directional image-brush mosaics (Phases C-F).

## Running the Demonstration Script

To run the automated sample generator:

```bash
# From the repository root (d:\Code\image-stroke-art)
py examples/generate_samples.py
```

This script will:
1. Synthesize a multi-region target painting (`sample_target.png`) containing a sunset sky gradient, a bright sun orb, mountain silhouettes, and lake reflections.
2. Build a curated library of textured source brush tiles (`sample_sources/`) across distinct color palettes.
3. Render the **Uniform Grid Photomosaic** (`output/grid_mosaic.png`).
4. Render the **Composition-Aware Adaptive Brush Mosaic** (`output/adaptive_brush_mosaic.png`) and its downscaled preview.
5. Print comparative diagnostics (PSNR, CIELAB Mean Delta E, stroke count, distinct sources, usage entropy).

## Running via the Command Line Interface (CLI)

You can also run the CLI directly on the sample files:

### Baseline Uniform-Grid Mode

```bash
py -m image_mosaic \
    --target examples/sample_target.png \
    --sources examples/sample_sources \
    --output examples/output/cli_grid.png \
    --mode grid \
    --tile-size 20 \
    --seed 42
```

### Composition-Aware Adaptive Brush Mode

```bash
py -m image_mosaic \
    --target examples/sample_target.png \
    --sources examples/sample_sources \
    --output examples/output/cli_adaptive.png \
    --preview examples/output/cli_adaptive_preview.png \
    --mode adaptive \
    --min-stroke-size 12 \
    --max-stroke-size 44 \
    --stroke-aspect-ratio 1.3 \
    --brush-mask organic_stroke \
    --color-correction reinhard \
    --color-strength 0.85 \
    --original-blend 0.15 \
    --directional \
    --reuse-penalty 0.3 \
    --seed 42
```

## Generated Outputs

- `sample_target.png`: Target composition (400x400)
- `output/grid_mosaic.png`: Uniform grid output (400 tiles)
- `output/adaptive_brush_mosaic.png`: Spatially adaptive multi-scale brush output (1,010 strokes)
- `output/adaptive_brush_mosaic_preview.png`: Downscaled preview
