from dataclasses import dataclass
from typing import Any

import torch

from project_main.tokens import Vocab


@dataclass
class Batch:
    """
    tokens:
        Input token IDs, shape [batch, seq_len]

    targets:
        Target token IDs, shape [batch, seq_len]
        targets[:, i] is the token the model should predict at position i.
    """
    tokens: torch.Tensor
    targets: torch.Tensor



# Extracts character count from token name
def extract_character_count(token_name: str) -> int:
    if token_name == "BOS" or token_name == "NEWLINE":
        return 0
    else:
        return int(token_name[-1])

#Inserts newlines into a sequence of token_ids
def insert_newlines(token_ids: list[int], vocab: Vocab, line_size: int) -> list[int]:
    current_line_len = 0
    output_token_ids = []

    for token_id in token_ids:
        token_name = vocab.decode_token(token_id)
        char_count = extract_character_count(token_name)
        current_line_len += char_count

        if current_line_len > line_size:
            # Insert newline token before the current token
            output_token_ids.append(vocab.newline_id)
            current_line_len = char_count  # Reset line length after newline
        output_token_ids.append(token_id)
    return output_token_ids


def generate_sequence(
    vocab: Vocab,
    task_cfg: dict,
    generator: torch.Generator | None = None,
) -> tuple[list[int], list[int]]:
    """
    Generate the tokens for one sequence.

    Returns:
        content_tokens:
            A list of length seq_len.
    """
    seq_len = task_cfg["seq_len"]

    # Non-special token IDs are the integers from 2 to vocab.size-1
    sampled_indices = torch.randint(
        low=2,
        high=vocab.size,
        size=(seq_len,),
        generator=generator,
    ).tolist()
    
    raw_token_sequence = [vocab.bos_id] + sampled_indices

    ##Insert new_lines as necessary
    line_size = task_cfg["line_size"]
    corrected_token_sequence = insert_newlines(raw_token_sequence, vocab, line_size)
    cropped_token_sequence = corrected_token_sequence[:seq_len + 1]

    input_tokens = cropped_token_sequence[:-1]
    target_tokens = cropped_token_sequence[1:]

    assert len(input_tokens) == seq_len
    assert len(target_tokens) == seq_len

    return input_tokens, target_tokens




def make_batch(
    batch_size: int,
    vocab: Vocab,
    task_cfg: dict,
    device: str,
    seed: int | None = None,
) -> Batch:
    """
    Generate a batch of next-token-prediction examples.
    """

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    all_tokens = []
    all_targets = []

    for example_idx in range(batch_size):
        tokens, targets = generate_sequence(
            vocab=vocab,
            task_cfg=task_cfg,
            generator=generator,
        )

        all_tokens.append(tokens)
        all_targets.append(targets)
       

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long, device=device)

    return Batch(
        tokens=tokens_tensor,
        targets=targets_tensor,
    )


def make_eval_batch(
    batch_size: int,
    vocab: Vocab,
    task_cfg: dict,
    device: str,
    fixed_eval_seed: int,
) -> Batch:
    """
    Generate a fixed evaluation batch.
    """

    return make_batch(
        batch_size=batch_size,
        vocab=vocab,
        task_cfg = task_cfg,
        device=device,
        seed=fixed_eval_seed,
    )


