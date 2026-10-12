from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = next(p for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent] if (p / "results/line_breaks_results").exists())
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import project_main.checkpoints as checkpoints
import project_main.data as data
import project_main.model as model_lib
import project_main.tokens as tokens_lib

def load_context(seed):
    run = PROJECT_ROOT / "results/line_breaks_results"
    cfg = json.loads((run / "config.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = tokens_lib.build_vocab(cfg["task"])
    model = model_lib.build_model(cfg=cfg, device=device)
    ckpt = checkpoints.load_checkpoint(path=run / "checkpoints/final.pt", model=model, map_location=device)
    model.eval()
    batch = data.make_batch(batch_size=1, vocab=vocab, task_cfg=cfg["task"], device=device, seed=seed)
    return type("Context", (), {"model": model, "vocab": vocab, "batch": batch, "cfg": cfg, "device": device, "checkpoint": ckpt})

def newline_positions(tokens, vocab):
    return [i for i, token in enumerate(tokens) if int(token) == vocab.newline_id]

def token_length(token_id, vocab):
    name = vocab.decode_token(int(token_id))
    return np.nan if name == "_NEWLINE_" else data.extract_character_count(name)

def cumulative_counts(tokens, vocab):
    out, current = [], 0
    for token in tokens:
        out.append(current)
        name = vocab.decode_token(int(token))
        if name == "_NEWLINE_": current = 0
        else: current += data.extract_character_count(name)
    return np.asarray(out, dtype=float)

def positions_since_newline(tokens, vocab):
    out, current = [], 0
    for token in tokens:
        out.append(current); current = 0 if int(token) == vocab.newline_id else current + 1
    return np.asarray(out, dtype=float)

def line_segments(tokens, vocab):
    boundaries = [-1] + newline_positions(tokens, vocab) + [len(tokens)]
    return [(boundaries[i] + 1, boundaries[i + 1]) for i in range(len(boundaries) - 1)]

def shared_pca(arrays):
    flat = np.concatenate(arrays, axis=0)
    pca = PCA(n_components=3, random_state=6000).fit(flat)
    return [pca.transform(a) for a in arrays]

def plot_trajectory(coords, labels, title, filename):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(111, projection="3d")
    for c, label in zip(coords, labels): ax.plot(c[:, 0], c[:, 1], c[:, 2], marker=".", linewidth=1.5, label=label)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(PROJECT_ROOT / filename, dpi=160); plt.close(fig)

def plot_heatmap(matrix, title, filename):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(matrix, origin="upper", aspect="auto", interpolation="nearest"); ax.set_title(title); ax.set_xlabel("Query position"); ax.set_ylabel("Key position"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(PROJECT_ROOT / filename, dpi=160); plt.close(fig)

def plot_scalar_by_position(values, labels, title, filename):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3)); ax.plot(values, marker="."); ax.set_title(title); ax.set_xlabel("Position"); ax.set_ylabel("Value"); fig.tight_layout(); fig.savefig(PROJECT_ROOT / filename, dpi=160); plt.close(fig)
