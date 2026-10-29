"""Command Line Interface for the Composition-Aware Image Brush Mosaic Generator."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from image_mosaic.config import (
    BrushMaskType,
    ColorCorrectionMode,
    MatchWeights,
    MosaicConfig,
    RenderingMode,
)
from image_mosaic.image_library import EmptyLibraryError, ImageLibrary
from image_mosaic.output import evaluate_quality, save_output
from image_mosaic.painting_analyzer import PaintingAnalyzer
from image_mosaic.renderer import MosaicRenderer
from image_mosaic.stroke_planner import AdaptiveStrokePlanner, GridStrokePlanner

app = typer.Typer(
    help="Reconstruct target paintings using collections of source images as visual brushstrokes.",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def generate(
    target: Path = typer.Option(
        ...,
        "--target",
        "-t",
        help="Path to the target painting or image to reconstruct.",
        exists=False,
    ),
    sources: Path = typer.Option(
        ...,
        "--sources",
        "-s",
        help="Directory containing source images to use as brushstrokes.",
        exists=False,
    ),
    output: Path = typer.Option(
        Path("mosaic_output.png"),
        "--output",
        "-o",
        help="Path where the final mosaic image will be saved.",
    ),
    preview: Optional[Path] = typer.Option(
        None,
        "--preview",
        "-p",
        help="Optional path to save a downscaled preview image.",
    ),
    mode: RenderingMode = typer.Option(
        RenderingMode.ADAPTIVE,
        "--mode",
        "-m",
        help="Rendering mode: 'grid' (uniform tiles) or 'adaptive' (multi-scale brushstrokes).",
    ),
    tile_size: int = typer.Option(
        24,
        "--tile-size",
        help="Tile dimension in pixels for grid mode.",
    ),
    min_stroke_size: int = typer.Option(
        12,
        "--min-stroke-size",
        help="Minimum brushstroke dimension in pixels for high-detail regions in adaptive mode.",
    ),
    max_stroke_size: int = typer.Option(
        48,
        "--max-stroke-size",
        help="Maximum brushstroke dimension in pixels for broad background regions in adaptive mode.",
    ),
    stroke_aspect_ratio: float = typer.Option(
        1.2,
        "--stroke-aspect-ratio",
        help="Aspect ratio (width / height) for elongated brushstrokes.",
    ),
    brush_mask: BrushMaskType = typer.Option(
        BrushMaskType.ORGANIC_STROKE,
        "--brush-mask",
        help="Stroke mask contour: 'organic_stroke', 'ellipse', or 'rectangle'.",
    ),
    color_correction: ColorCorrectionMode = typer.Option(
        ColorCorrectionMode.REINHARD,
        "--color-correction",
        help="Color adaptation mode: 'reinhard', 'mean_shift', 'tint', or 'none'.",
    ),
    color_strength: float = typer.Option(
        0.85,
        "--color-strength",
        help="Strength of color correction (0.0 = original source colors, 1.0 = full target adaptation).",
    ),
    luminance_preservation: float = typer.Option(
        0.5,
        "--luminance-preservation",
        help="Preservation of source image internal contrast and luminance detail (0.0 to 1.0).",
    ),
    original_blend: float = typer.Option(
        0.15,
        "--original-blend",
        help="Blend strength of original painting into final mosaic (0.0 = pure strokes, 1.0 = original).",
    ),
    directional: bool = typer.Option(
        True,
        "--directional/--no-directional",
        help="Align strokes along local edge contours and tangents.",
    ),
    max_rotation: float = typer.Option(
        90.0,
        "--max-rotation",
        help="Maximum stroke rotation in degrees.",
    ),
    overlap_ratio: float = typer.Option(
        0.25,
        "--overlap-ratio",
        help="Overlap ratio between adjacent brushstrokes to eliminate gaps.",
    ),
    max_reuse: int = typer.Option(
        0,
        "--max-reuse",
        help="Maximum allowed usage count per source image (0 = unlimited).",
    ),
    reuse_penalty: float = typer.Option(
        0.3,
        "--reuse-penalty",
        help="Penalty factor discouraging repeated use of the same source images.",
    ),
    w_color: float = typer.Option(1.0, "--w-color", help="Matching weight for mean Lab color."),
    w_luminance: float = typer.Option(0.5, "--w-luminance", help="Matching weight for luminance stats."),
    w_hist: float = typer.Option(0.3, "--w-hist", help="Matching weight for color histogram."),
    w_texture: float = typer.Option(0.2, "--w-texture", help="Matching weight for texture and gradient."),
    seed: Optional[int] = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible layout and tie-breaking.",
    ),
    max_resolution: Optional[int] = typer.Option(
        None,
        "--max-resolution",
        help="Optionally downscale target image if max dimension exceeds this value.",
    ),
    refinement_passes: int = typer.Option(
        1,
        "--refinement-passes",
        help="Number of multi-scale composition refinement passes.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose debug logging.",
    ),
) -> None:
    """Generate a composition-aware image brush mosaic."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Sanitize path inputs against Windows shell escaping artifacts (e.g. \" at end of string)
    target = Path(str(target).strip().strip('"\''))
    sources = Path(str(sources).strip().strip('"\''))
    output = Path(str(output).strip().strip('"\''))
    if preview is not None:
        preview = Path(str(preview).strip().strip('"\''))

    # Construct configuration object
    config = MosaicConfig(
        target_path=target,
        sources_dir=sources,
        output_path=output,
        preview_path=preview,
        mode=mode,
        tile_size=tile_size,
        min_stroke_size=min_stroke_size,
        max_stroke_size=max_stroke_size,
        stroke_aspect_ratio=stroke_aspect_ratio,
        brush_mask=brush_mask,
        color_correction=color_correction,
        color_strength=color_strength,
        luminance_preservation=luminance_preservation,
        original_blend=original_blend,
        enable_directional=directional,
        max_rotation_degrees=max_rotation,
        overlap_ratio=overlap_ratio,
        max_reuse_count=max_reuse,
        reuse_penalty_weight=reuse_penalty,
        weights=MatchWeights(
            color=w_color,
            luminance=w_luminance,
            histogram=w_hist,
            texture=w_texture,
            reuse=reuse_penalty,
        ),
        seed=seed,
        max_resolution=max_resolution,
        refinement_passes=refinement_passes,
        verbose=verbose,
    )

    try:
        config.validate()
    except (FileNotFoundError, NotADirectoryError, ValueError) as err:
        err_console.print(f"[bold red]Configuration Error:[/bold red] {err}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Composition-Aware Image Brush Mosaic Generator[/bold cyan]\n"
            f"Target: [green]{target}[/green]\n"
            f"Sources: [green]{sources}[/green]\n"
            f"Mode: [yellow]{mode.value}[/yellow] | Seed: {seed}",
            title="Image Mosaic",
        )
    )

    t0 = time.time()

    # Step 1: Load and index source image library
    try:
        with console.status("[bold green]Loading source image library..."):
            library = ImageLibrary(sources_dir=sources)
            library.load()
        console.print(f"[green][OK][/green] Loaded [bold]{library.count}[/bold] unique source images.")
    except EmptyLibraryError as err:
        err_console.print(f"[bold red]Library Error:[/bold red] {err}")
        raise typer.Exit(code=1)
    except Exception as err:
        err_console.print(f"[bold red]Error loading library:[/bold red] {err}")
        raise typer.Exit(code=1)

    # Step 2: Analyze target image
    try:
        with console.status("[bold green]Analyzing target composition, gradients & saliency..."):
            analyzer = PaintingAnalyzer(
                image_or_path=target,
                max_resolution=max_resolution,
            )
        console.print(
            f"[green][OK][/green] Analyzed target image ({analyzer.width}x{analyzer.height})."
        )
    except Exception as err:
        err_console.print(f"[bold red]Error analyzing target image:[/bold red] {err}")
        raise typer.Exit(code=1)

    # Step 3: Plan strokes
    with console.status(f"[bold green]Planning strokes ({mode.value} mode)..."):
        if mode == RenderingMode.GRID:
            planner = GridStrokePlanner(config)
        else:
            planner = AdaptiveStrokePlanner(config)
        plan = planner.plan(analyzer, library)
    console.print(f"[green][OK][/green] Generated stroke plan with [bold]{plan.total_strokes}[/bold] strokes.")

    # Step 4: Render canvas
    with console.status("[bold green]Compositing visual brushstrokes..."):
        renderer = MosaicRenderer(config)
        rendered_image = renderer.render(plan, library, analyzer)
    console.print("[green][OK][/green] Rendering and multi-scale refinement complete.")

    # Step 5: Save output & compute metrics
    try:
        with console.status("[bold green]Saving output and evaluating metrics..."):
            saved_output_path = save_output(
                rendered_image,
                output_path=output,
                preview_path=preview,
            )
            metrics = evaluate_quality(analyzer.target_image, rendered_image, plan)
    except Exception as err:
        err_console.print(f"[bold red]Error saving output image:[/bold red] {err}")
        raise typer.Exit(code=1)

    elapsed = round(time.time() - t0, 2)
    console.print(f"[green][OK][/green] Output successfully saved to [bold]{saved_output_path}[/bold] (in {elapsed}s).")

    # Display diagnostics table
    table = Table(title="Mosaic Diagnostics & Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Total Strokes Placed", str(metrics.total_strokes))
    table.add_row("Distinct Source Images Used", str(metrics.distinct_sources_used))
    table.add_row("Max Single-Source Reuse", str(metrics.max_source_reuse))
    table.add_row("Source Usage Entropy", f"{metrics.repetition_entropy} bits")
    table.add_row("CIELAB Mean Delta E", f"{metrics.lab_delta_e_mean:.2f}")
    table.add_row("PSNR (fidelity)", f"{metrics.psnr_db:.2f} dB")

    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
