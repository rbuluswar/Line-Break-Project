from project_main.train import train


def test_train_smoke(tmp_path):
    cfg = {
        "experiment": {
            "name": "smoke_test",
            "seed": 0,
            "device": "cpu",
        },
        "task": {
            "name": "test_task",
            "seq_len": 8,
            "vocab_size": 30,
            "line_size": 10,
        },
        "data": {
            "batch_size": 2,
            "eval_batch_size": 2,
            "fixed_eval_seed": 123,
        },
        "model": {
            "n_layers": 1,
            "n_heads": 1,
            "d_model": 32,
            "d_head": 32,
            "d_mlp": 64,
            "act_fn": "gelu",
            "normalization_type": "LN",
        },
        "optimizer": {
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "name": "constant",
        },
        "training": {
            "num_steps": 2,
            "grad_clip_norm": 1.0,
        },
        "logging": {
            "log_every": 1,
            "eval_every": 1,
            "save_every": 1,
        },
        "paths": {
            "output_dir": str(tmp_path / "results"),
        },
    }

    train(cfg)