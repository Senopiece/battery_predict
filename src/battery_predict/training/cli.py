"""Training CLI using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from battery_predict.training.config import load_experiment_config


def set_nested_attr(obj, key_path, value):
    """Set a nested attribute on a dataclass using dot notation."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)


from battery_predict.training.orchestration import fit_experiment

app = typer.Typer(help="Train the battery latent capacity predictor.")


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        "configs/default.yaml",
        "--config",
        help="Path to experiment config YAML.",
    ),
    overrides: list[str] = typer.Argument(
        None,
        help="Override config entries with key=value (dot notation for nested). Example: --config.data.dataset_dir=data2/set",
    ),
) -> None:
    """Train the model.

    Example:
        train
        train --config configs/custom.yaml
        train --config.data.dataset_dir=data2/set
    """
    loaded_config = load_experiment_config(config)

    # Parse and apply overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            # Try to infer type from current value
            try:
                current = loaded_config
                for k in key.split("."):
                    current = getattr(current, k)
                # Try to cast value to type of current
                if isinstance(current, bool):
                    value = value.lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, tuple):
                    value = tuple(type(current[0])(v) for v in value.split(","))
            except Exception:
                pass
            set_nested_attr(loaded_config, key, value)

    trainer, _, _, run_dir = fit_experiment(loaded_config, enable_live_plot=False)

    # Save the final config (with overrides) to run_dir/config.yaml for ClearML
    loaded_config.save_yaml(run_dir / "config.yaml")

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
