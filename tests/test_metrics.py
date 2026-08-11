import pytest
import torch

from project_main.metrics import (
    accuracy_when_prediction_is_from_first_target,
    accuracy_when_target_is_from_first_target,
    at_or_after_first_target_token_mask,
    special_token_metrics,
)


def _logits_from_predictions(predictions: torch.Tensor, vocab_size: int = 3) -> torch.Tensor:
    return torch.nn.functional.one_hot(predictions, num_classes=vocab_size).float()


def test_first_target_mask_is_computed_per_sequence() -> None:
    targets = torch.tensor(
        [
            [0, 1, 2, 1],
            [1, 0, 2, 0],
            [0, 2, 0, 2],
        ]
    )

    mask = at_or_after_first_target_token_mask(targets, token_id=1)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [True, True, True, True],
                [False, False, False, False],
            ]
        ),
    )


def test_prediction_accuracy_ignores_positions_before_first_true_target() -> None:
    targets = torch.tensor(
        [
            [0, 0, 1, 0],
            [0, 2, 0, 2],
        ]
    )
    predictions = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )

    accuracy = accuracy_when_prediction_is_from_first_target(
        _logits_from_predictions(predictions),
        targets,
        token_id=1,
    )

    # Only the final two positions in the first sequence are eligible. The
    # second sequence is excluded because its targets never contain token 1.
    assert accuracy == pytest.approx(0.5)


def test_target_accuracy_includes_first_true_target() -> None:
    targets = torch.tensor([[0, 1, 0, 1]])
    predictions = torch.tensor([[1, 1, 0, 0]])

    accuracy = accuracy_when_target_is_from_first_target(
        _logits_from_predictions(predictions),
        targets,
        token_id=1,
    )

    assert accuracy == pytest.approx(0.5)


def test_special_token_metrics_exposes_from_first_target_accuracies() -> None:
    targets = torch.tensor([[0, 1, 0]])
    predictions = torch.tensor([[1, 1, 1]])

    metrics = special_token_metrics(
        _logits_from_predictions(predictions),
        targets,
        token_id=1,
        token_name="newline",
    )

    assert metrics["accuracy_when_target_is_newline_from_first_newline"] == 1.0
    assert metrics["accuracy_when_prediction_is_newline_from_first_newline"] == 0.5
