"""Tests for CLI argument validation and execution."""

from __future__ import annotations

from pathlib import Path
from typer.testing import CliRunner
import pytest

from image_mosaic.cli import app

runner = CliRunner()


def test_cli_missing_required_args():
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    # Output should contain missing option indicator
    assert "Missing option" in result.output or "Error" in result.output or "--target" in result.output


def test_cli_nonexistent_target(tmp_path: Path):
    nonexistent = tmp_path / "not_there.png"
    sources = tmp_path / "sources"
    sources.mkdir()

    result = runner.invoke(app, ["--target", str(nonexistent), "--sources", str(sources)])
    assert result.exit_code == 1
    assert "Configuration Error" in result.output or "Target image not found" in result.output


def test_cli_full_execution_grid(synthetic_target: Path, synthetic_sources_dir: Path, tmp_path: Path):
    out_file = tmp_path / "cli_grid_mosaic.png"
    result = runner.invoke(
        app,
        [
            "--target", str(synthetic_target),
            "--sources", str(synthetic_sources_dir),
            "--output", str(out_file),
            "--mode", "grid",
            "--tile-size", "25",
            "--seed", "42",
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    assert "Diagnostics & Metrics" in result.output
