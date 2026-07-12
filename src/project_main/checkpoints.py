from pathlib import Path
from typing import Any

import json
import torch
from transformer_lens import HookedTransformer


def save_config(cfg: dict[str, Any], output_dir: str | Path) -> Path:
    """
    Save the run config into the output directory.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "config.json"

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

    return path


def save_checkpoint(
    model: HookedTransformer,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
    output_dir: str | Path,
    scheduler: Any | None = None,
) -> Path:
    """
    Save model/training state at a particular training step.
    """

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"step_{step:06d}.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)

    return path


def save_final_checkpoint(
    model: HookedTransformer,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    step: int,
    output_dir: str | Path,
    scheduler: Any | None = None,
) -> Path:
    """
    Save a final checkpoint with a stable filename.
    """

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / "final.pt"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)

    return path


def load_checkpoint(
    path: str | Path,
    model: HookedTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint into an existing model.

    Optionally also loads optimizer and scheduler state.

    Returns:
        The full checkpoint dictionary.
    """

    path = Path(path)

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """
    Return the most recent step checkpoint in output_dir/checkpoints.

    Returns None if no step checkpoints exist.
    """

    checkpoint_dir = Path(output_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        return None

    paths = sorted(checkpoint_dir.glob("step_*.pt"))

    if not paths:
        return None

    return paths[-1]