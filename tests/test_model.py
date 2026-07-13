import torch

from project_main.model import build_model
from project_main.tokens import build_vocab



def test_model_forward_shape():
    cfg = {
        "task": {
            "seq_len": 8,
            "vocab_size": 30,
        },
        "model": {
            "n_layers": 2,
            "n_heads": 2,
            "d_model": 64,
            "d_head": 32,
            "d_mlp": 128,
            "act_fn": "gelu",
            "normalization_type": "LN",
        },
    }

    vocab = build_vocab(cfg["task"])

    model = build_model(
        cfg=cfg,
        device="cpu",
    )

    tokens = torch.randint(
        low=0,
        high=vocab.size,
        size=(4, cfg["task"]["seq_len"]),
    )

    logits = model(tokens)

    assert logits.shape == (4, cfg["task"]["seq_len"], vocab.size)

