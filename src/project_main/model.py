from typing import Any

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


def count_parameters(model: HookedTransformer) -> int:
    """
    Count trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_model_config(
    cfg: dict[str, Any],
) -> None:
    """
    Validate model-related config values.

    This catches common mistakes before model construction.
    """

    model_cfg = cfg["model"]
    seq_len = cfg["task"]["seq_len"]
    vocab_size = cfg["task"]["vocab_size"]    

    required_keys = [
        "n_layers",
        "n_heads",
        "d_model",
        "d_head",
        "d_mlp",
    ]

    for key in required_keys:
        if key not in model_cfg:
            raise KeyError(f"Missing model config key: model.{key}")

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    positive_keys = [
        "n_layers",
        "n_heads",
        "d_model",
        "d_head",
        "d_mlp",
    ]

    for key in positive_keys:
        if model_cfg[key] <= 0:
            raise ValueError(f"model.{key} must be positive, got {model_cfg[key]}")

    expected_d_model = model_cfg["n_heads"] * model_cfg["d_head"]

    if model_cfg["d_model"] != expected_d_model:
        raise ValueError(
            "For this starter setup, expected model.d_model to equal "
            "model.n_heads * model.d_head. "
            f"Got d_model={model_cfg['d_model']}, "
            f"n_heads={model_cfg['n_heads']}, "
            f"d_head={model_cfg['d_head']}, "
            f"n_heads*d_head={expected_d_model}."
        )



def build_model(
    cfg: dict[str, Any],
    device: str,
) -> HookedTransformer:
    """
    Build a TransformerLens HookedTransformer from the YAML config.

    Args:
        cfg:
            Full config dictionary loaded from YAML.
        device:
            "cuda" or "cpu".

    Returns:
        A HookedTransformer model.
    """

    model_cfg = cfg["model"]
    task_cfg = cfg["task"]  

    tl_cfg = HookedTransformerConfig(
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        d_head=model_cfg["d_head"],
        d_mlp=model_cfg["d_mlp"],
        d_vocab=task_cfg["vocab_size"],
        n_ctx=task_cfg["seq_len"],
        act_fn=model_cfg.get("act_fn", "gelu"),
        normalization_type=model_cfg.get("normalization_type", "LN"),
        device=device,
    )

    model = HookedTransformer(tl_cfg)

    return model