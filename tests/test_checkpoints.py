import torch

from project_main.checkpoints import save_checkpoint, load_checkpoint
from project_main.model import build_model
from project_main.tokens import build_vocab


def test_save_and_load_checkpoint(tmp_path):
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
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
        },
    }

    vocab = build_vocab(cfg["task"])

    model = build_model(cfg, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        step=10,
        output_dir=tmp_path,
    )

    assert path.exists()

    new_model = build_model(cfg, device="cpu")

    checkpoint = load_checkpoint(
        path=path,
        model=new_model,
        map_location="cpu",
    )

    assert checkpoint["step"] == 10