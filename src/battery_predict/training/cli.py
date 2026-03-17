from __future__ import annotations

import argparse
import sys

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
    import __main__

    # Fix for Windows: DataLoader workers (num_workers > 0) use Python's
    # multiprocessing 'spawn', which re-runs __main__.__file__ to recreate the
    # main module context. Console-script .exe launchers on Windows set argv[0]
    # (and therefore __main__.__file__) to the script path WITHOUT a .py extension
    # (e.g. battery-predict-train), which Python cannot open as a module file.
    # Pointing __main__.__file__ to this .py file (which only defines functions)
    # lets spawn import it safely without re-triggering training.
    if not getattr(__main__, "__file__", "").endswith(".py"):
        __main__.__file__ = __file__
        __main__.__spec__ = None

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
