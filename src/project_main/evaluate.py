from typing import Any

import torch

from project_main.data import make_eval_batch
from project_main.metrics import loss_fn, accuracy


@torch.no_grad()
def evaluate_model(
    model,
    vocab,
    cfg: dict[str, Any],
    device: str,
) -> dict[str, float]:
    model.eval()

    batch = make_eval_batch(
        batch_size=cfg["data"]["eval_batch_size"],
        seq_len=cfg["task"]["seq_len"],
        vocab=vocab,
        task_cfg=cfg["task"],
        device=device,
        fixed_eval_seed=cfg["data"]["fixed_eval_seed"],
    )

    logits = model(batch.tokens)

    metrics = {
        "eval_loss": loss_fn(logits, batch.targets).item(),
        "eval_accuracy": accuracy(logits, batch.targets),
    }

    model.train()

    return metrics