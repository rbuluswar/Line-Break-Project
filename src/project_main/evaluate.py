from typing import Any

import torch

from project_main.data import make_eval_batch
from project_main.metrics import loss_fn, accuracy, special_token_metrics


@torch.no_grad()
def evaluate_model(
    model,
    vocab,
    cfg: dict[str, Any],
    device: str,
    in_training: bool = True
) -> dict[str, float]:
    model.eval()

    batch = make_eval_batch(
        batch_size=cfg["data"]["eval_batch_size"],
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

    newline_metrics = special_token_metrics(
    logits=logits,
    targets=batch.targets,
    token_id=vocab.newline_id,
    token_name="newline",
)

    metrics.update({f"eval_{k}": v for k, v in newline_metrics.items()})

    if in_training:
        model.train()

    return metrics