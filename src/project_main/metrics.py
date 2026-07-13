import torch
import torch.nn.functional as F


def loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute next-token prediction cross-entropy loss.

    Args:
        logits:
            Tensor of shape [batch, seq_len, vocab_size].
        targets:
            Tensor of shape [batch, seq_len].
        pad_id:
            Optional token ID to ignore in the loss.

    Returns:
        Scalar loss tensor.
    """

    vocab_size = logits.size(-1)

    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    return F.cross_entropy(flat_logits, flat_targets)



@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        logits:
            Tensor of shape [batch, seq_len, vocab_size].
        targets:
            Tensor of shape [batch, seq_len].
    Returns:
        Accuracy as a Python float.
    """

    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()



@torch.no_grad()
def logit_at_token(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    position: int = -1,
) -> torch.Tensor:
    """
    Get the logit assigned to a particular token at a particular position.

    Args:
        logits:
            [batch, seq_len, vocab_size]
        token_ids:
            [batch]
        position:
            Sequence position to inspect.

    Returns:
        Tensor of shape [batch].
    """

    position_logits = logits[:, position, :]
    return position_logits.gather(dim=-1, index=token_ids[:, None]).squeeze(-1)


@torch.no_grad()
def logit_diff(
    logits: torch.Tensor,
    correct_tokens: torch.Tensor,
    incorrect_tokens: torch.Tensor,
    position: int = -1,
) -> torch.Tensor:
    """
    Compute logit(correct) - logit(incorrect) at a given position.

    Useful for clean/corrupted activation patching.
    """

    correct_logits = logit_at_token(logits, correct_tokens, position=position)
    incorrect_logits = logit_at_token(logits, incorrect_tokens, position=position)

    return correct_logits - incorrect_logits


@torch.no_grad()
def masked_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Compute accuracy only over positions where mask is True.

    Args:
        logits:
            Tensor of shape [batch, seq_len, vocab_size].
        targets:
            Tensor of shape [batch, seq_len].
        mask:
            Boolean tensor of shape [batch, seq_len].

    Returns:
        Accuracy over masked positions as a Python float.
        Returns 0.0 if the mask selects no positions.
    """
    preds = logits.argmax(dim=-1)

    if mask.sum().item() == 0:
        return 0.0

    return (preds[mask] == targets[mask]).float().mean().item()



@torch.no_grad()
def accuracy_when_target_is(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_id: int,
) -> float:
    """
    Accuracy only at positions where the correct next token is token_id.

    Example:
        accuracy when the target token is NEWLINE.
    """

    mask = targets == token_id
    return masked_accuracy(logits, targets, mask)



@torch.no_grad()
def accuracy_when_target_is_not(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_id: int,
) -> float:
    """
    Accuracy only at positions where the correct next token is not token_id.

    Example:
        accuracy when the target token is not NEWLINE.
    """

    mask = targets != token_id
    return masked_accuracy(logits, targets, mask)


@torch.no_grad()
def target_token_fraction(
    targets: torch.Tensor,
    token_id: int,
) -> float:
    """
    Fraction of valid target positions equal to token_id.
    """

    valid_mask = torch.ones_like(targets, dtype=torch.bool)

    if valid_mask.sum().item() == 0:
        return 0.0

    token_mask = targets == token_id

    return (token_mask & valid_mask).float().sum().item() / valid_mask.float().sum().item()


@torch.no_grad()
def target_token_prediction_frequency(
    logits: torch.Tensor,
    token_id: int,
) -> float:
    """
    Fraction of predictions equal to token_id.
    """
    preds = logits.argmax(dim=-1)
    return (preds == token_id).float().mean().item()



@torch.no_grad()
def special_token_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_id: int,
    token_name: str,
) -> dict[str, float]:
    """
    Return metrics split by whether the target is a particular token.
    """

    return {
        f"accuracy_when_target_is_{token_name}": accuracy_when_target_is(
            logits=logits,
            targets=targets,
            token_id=token_id,
        ),
        f"accuracy_when_target_is_not_{token_name}": accuracy_when_target_is_not(
            logits=logits,
            targets=targets,
            token_id=token_id,
        ),
        f"target_fraction_{token_name}": target_token_fraction(
            targets=targets,
            token_id=token_id,
        ),
        f"target_prediction_frequency_{token_name}": target_token_prediction_frequency(
            logits=logits,
            token_id=token_id,
        ),
    }
