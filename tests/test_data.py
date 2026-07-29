import torch

import project_main.data as project_data
from project_main.data import (
    get_line_length_range,
    make_batch,
    make_eval_batch,
    sample_line_length,
)
from project_main.tokens import build_vocab
from project_main.data import insert_newlines


def test_batch_shapes():
    cfg = {
        "vocab_size": 30,
        "seq_len": 8,
        "min_line_length": 5,
        "max_line_length": 10,
        "num_token_lengths": 4,
    }
    vocab = build_vocab(cfg)

    batch = make_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
    )

    assert batch.tokens.shape == (4, 8)
    assert batch.targets.shape == (4, 8)
    assert batch.tokens.dtype == torch.long
    assert batch.targets.dtype == torch.long


def test_targets_are_shifted_inputs():
    cfg = {
        "vocab_size": 30,
        "seq_len": 8,
        "min_line_length": 5,
        "max_line_length": 10,
        "num_token_lengths": 4,
    }
    vocab = build_vocab(cfg)

    batch = make_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
        seed=0,
    )

    assert torch.equal(batch.tokens[:, 1:], batch.targets[:, :-1])


def test_eval_batch_is_repeatable():
    cfg = {
        "vocab_size": 30,
        "seq_len": 8,
        "min_line_length": 5,
        "max_line_length": 10,
        "num_token_lengths": 4,
    }
    vocab = build_vocab(cfg)

    batch1 = make_eval_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
        fixed_eval_seed=123,
    )

    batch2 = make_eval_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
        fixed_eval_seed=123,
    )

    assert torch.equal(batch1.tokens, batch2.tokens)
    assert torch.equal(batch1.targets, batch2.targets)


def test_training_batches_change_without_seed():
    cfg = {
        "vocab_size": 30,
        "seq_len": 8,
        "min_line_length": 5,
        "max_line_length": 10,
        "num_token_lengths": 4,
    }
    vocab = build_vocab(cfg)

    batch1 = make_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
    )

    batch2 = make_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
    )

    assert not torch.equal(batch1.tokens, batch2.tokens)


def test_line_length_is_sampled_uniformly_from_inclusive_range():
    cfg = {"min_line_length": 3, "max_line_length": 5}
    generator = torch.Generator().manual_seed(0)

    samples = [
        sample_line_length(cfg, generator=generator)
        for _ in range(6_000)
    ]
    counts = torch.bincount(torch.tensor(samples), minlength=6)

    assert set(samples) == {3, 4, 5}
    assert torch.all((counts[3:6] > 1_800) & (counts[3:6] < 2_200))


def test_make_batch_draws_one_line_length_per_example(monkeypatch):
    cfg = {
        "vocab_size": 30,
        "seq_len": 8,
        "min_line_length": 5,
        "max_line_length": 10,
        "num_token_lengths": 4,
    }
    vocab = build_vocab(cfg)
    sampled_line_lengths = []
    original_sampler = project_data.sample_line_length

    def recording_sampler(task_cfg, generator=None):
        line_length = original_sampler(task_cfg, generator=generator)
        sampled_line_lengths.append(line_length)
        return line_length

    monkeypatch.setattr(project_data, "sample_line_length", recording_sampler)

    make_batch(
        batch_size=4,
        vocab=vocab,
        task_cfg=cfg,
        device="cpu",
        seed=0,
    )

    assert len(sampled_line_lengths) == 4
    assert all(5 <= line_length <= 10 for line_length in sampled_line_lengths)


def test_line_length_range_validation():
    assert get_line_length_range(
        {"min_line_length": 5, "max_line_length": 10}
    ) == (5, 10)

    invalid_configs = [
        {"min_line_length": 0, "max_line_length": 10},
        {"min_line_length": 11, "max_line_length": 10},
        {"min_line_length": 5},
        {"max_line_length": 10},
        {"min_line_length": 5.0, "max_line_length": 10},
    ]
    for cfg in invalid_configs:
        try:
            get_line_length_range(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid line-length config: {cfg}")


def test_legacy_line_size_config_is_supported():
    assert get_line_length_range({"line_size": 10}) == (10, 10)



def test_newline_inserted_before_line_overflow():
    cfg = {"vocab_size": 30, "num_token_lengths": 4}
    vocab = build_vocab(cfg)

    token_ids = [
        vocab.encode_token("TOKEN_1_3"),
        vocab.encode_token("TOKEN_2_3"),
        vocab.encode_token("TOKEN_3_4"),
    ]

    wrapped = insert_newlines(
        token_ids=token_ids,
        vocab=vocab,
        line_size=6,
    )

    assert vocab.decode(wrapped) == [
        "TOKEN_1_3",
        "TOKEN_2_3",
        "NEWLINE",
        "TOKEN_3_4",
    ]
