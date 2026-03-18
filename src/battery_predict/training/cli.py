"""Training CLI using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from battery_predict.training.config import load_experiment_config
from battery_predict.training.orchestration import fit_experiment

app = typer.Typer(help="Train the battery latent capacity predictor.")


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to experiment config YAML. If None, uses built-in defaults.",
    ),
) -> None:
    """Train the model with optional config override.

    Example:
        train --config configs/default.yaml
    """
    loaded_config = load_experiment_config(config)
    trainer, _, _, run_dir = fit_experiment(loaded_config, enable_live_plot=False)

    # Find and display best checkpoint
    checkpoint_callback = next(
        (cb for cb in trainer.callbacks if hasattr(cb, "best_model_path")),
        None,
    )

    typer.echo(f"\n✓ Training complete")
    typer.echo(f"  Run directory:   {run_dir}")
    if checkpoint_callback:
        typer.echo(f"  Best checkpoint: {checkpoint_callback.best_model_path}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
