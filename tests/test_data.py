import torch

from project_main.data import make_batch, make_eval_batch
from project_main.tokens import build_vocab
from project_main.data import insert_newlines


def test_batch_shapes():
    cfg = {"vocab_size": 30, "seq_len": 8, "line_size": 10}
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
    cfg = {"vocab_size": 30, "seq_len": 8, "line_size": 10}
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
    cfg = {"vocab_size": 30, "seq_len": 8, "line_size": 10}
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
    cfg = {"vocab_size": 30, "seq_len": 8, "line_size": 10}
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



def test_newline_inserted_before_line_overflow():
    cfg = {"vocab_size": 30}
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

    assert vocab.decode(wrapped) == ["TOKEN_1_3", "TOKEN_2_3", "\n", "TOKEN_3_4"]