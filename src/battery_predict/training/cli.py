from __future__ import annotations

import argparse

from battery_predict.training.config import load_experiment_config
from battery_predict.training.run import fit_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the battery latent predictor.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a YAML config file."
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the test pass after fitting.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    trainer, _, _, run_dir = fit_experiment(
        config,
        enable_live_plot=False,
        run_test=not args.skip_test,
    )
    checkpoint_callback = next(
        callback
        for callback in trainer.callbacks
        if hasattr(callback, "best_model_path")
    )
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
