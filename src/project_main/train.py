
## IMPORTS

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm


from project_main.checkpoints import (
    save_config,
    save_checkpoint,
    save_final_checkpoint,
)
from project_main.data import make_batch
from project_main.evaluate import evaluate_model
from project_main.metrics import (
    loss_fn,
    accuracy,
    special_token_metrics,
)
from project_main.model import build_model, count_parameters
from project_main.tokens import build_vocab


## HELPER FUNCTIONS
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def resolve_device(cfg: dict[str, Any]) -> str:
    """
    Decide whether to use CPU or CUDA.

    If config says cuda but CUDA is unavailable, raise an error.
    """

    requested_device = cfg["experiment"].get("device", "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested device='cuda', but torch.cuda.is_available() is False."
        )

    return requested_device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    """

    opt_cfg = cfg["optimizer"]
    name = opt_cfg["name"].lower()

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
            eps=opt_cfg.get("eps", 1e-8),
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )

    raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")



def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
):
    """
    Build learning-rate scheduler from config.
    """

    scheduler_cfg = cfg.get("scheduler", {"name": "constant"})
    name = scheduler_cfg.get("name", "constant").lower()

    if name == "constant":
        return None

    if name == "cosine":
        num_steps = cfg["training"]["num_steps"]
        warmup_steps = scheduler_cfg.get("warmup_steps", 0)
        min_lr = scheduler_cfg.get("min_lr", 0.0)
        base_lr = cfg["optimizer"]["lr"]

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            if num_steps == warmup_steps:
                return 1.0

            progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)

            cosine_factor = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
            min_lr_factor = min_lr / base_lr

            return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unknown scheduler: {scheduler_cfg['name']}")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from the first optimizer param group.
    """

    return optimizer.param_groups[0]["lr"]



### MAIN TRAINING FUNCTION
def train(cfg: dict[str, Any]) -> None:
    """
    Run a full training job.
    """

    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    device = resolve_device(cfg)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir)

    vocab = build_vocab(cfg["task"])

    model = build_model(
        cfg=cfg,
        device=device,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    print(f"Device: {device}")
    print(f"Vocab size: {vocab.size}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Output directory: {output_dir}")

    num_steps = cfg["training"]["num_steps"]
    batch_size = cfg["data"]["batch_size"]
    seq_len = cfg["task"]["seq_len"]

    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["logging"]["save_every"]

    model.train()

    progress = tqdm(range(1, num_steps + 1), desc="training")

    for step in progress:
        batch = make_batch(
            batch_size=batch_size,
            vocab=vocab,
            task_cfg=cfg["task"],
            device=device,
        )

        logits = model(batch.tokens)

        loss = loss_fn(
            logits=logits,
            targets=batch.targets,
        )

        optimizer.zero_grad()
        loss.backward()

        grad_clip_norm = cfg["training"].get("grad_clip_norm")
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0 or step == 1:
            train_acc = accuracy(
                logits=logits,
                targets=batch.targets,
            )


            metrics = {
                "step": step,
                "train_loss": loss.item(),
                "train_accuracy": train_acc,
                "lr": get_current_lr(optimizer),
            }

            # Newline-specific metrics
            newline_metrics = special_token_metrics(
                    logits=logits,
                    targets=batch.targets,
                    token_id=vocab.newline_id,
                    token_name="newline",
                )
            metrics.update({f"train_{k}": v for k, v in newline_metrics.items()})

            progress.set_postfix(
                loss=f"{metrics['train_loss']:.4f}",
                acc=f"{metrics['train_accuracy']:.4f}",
                lr=f"{metrics['lr']:.2e}",
            )

            print(metrics)

        if step % eval_every == 0 or step == 1:
            eval_metrics = evaluate_model(
                model=model,
                vocab=vocab,
                cfg=cfg,
                device=device,
            )

            eval_metrics = {
                "step": step,
                **eval_metrics,
            }

            print(eval_metrics)

        if step % save_every == 0:
            path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                step=step,
                output_dir=output_dir,
            )

            print(f"Saved checkpoint to {path}")

    final_path = save_final_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        step=num_steps,
        output_dir=output_dir,
    )

    print(f"Saved final checkpoint to {final_path}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()