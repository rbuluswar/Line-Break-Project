from pathlib import Path
import nbformat as nbf

out = Path(__file__).with_name("01_count_emergence_and_mlp_accumulation.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

setup = '''from pathlib import Path
import sys, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

ROOT = next(p for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent] if (p / "results/line_breaks_results").exists())
sys.path.insert(0, str(ROOT / "src"))
import project_main.checkpoints as checkpoints
import project_main.data as data
import project_main.model as model_lib
import project_main.tokens as tokens_lib

TRAIN_SEED, EVAL_SEED = 8011, 8012
N_TRAIN = N_EVAL = 256
BATCH_SIZE = 32
SITES = ["blocks.1.hook_resid_pre", "blocks.1.hook_resid_mid", "blocks.1.hook_mlp_out", "blocks.1.hook_resid_post"]
short = dict(zip(SITES, ["pre layer 1", "pre MLP", "MLP output", "post layer 1"]))
cfg = json.loads((ROOT / "results/line_breaks_results/config.json").read_text())
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab = tokens_lib.build_vocab(cfg["task"])
model = model_lib.build_model(cfg=cfg, device=device)
checkpoint = checkpoints.load_checkpoint(path=ROOT / "results/line_breaks_results/checkpoints/final.pt", model=model, map_location=device)
model.eval()

def collect(n, seed):
    toks, acts, labs = [], {s: [] for s in SITES}, []
    for start in range(0, n, BATCH_SIZE):
        size = min(BATCH_SIZE, n - start)
        batch = data.make_batch(size, vocab, cfg["task"], device, seed + start)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch.tokens, prepend_bos=False, names_filter=SITES)
        t = batch.tokens.detach().cpu().numpy(); toks.append(t)
        for s in SITES: acts[s].append(cache[s].detach().cpu().numpy())
        y = np.zeros(t.shape, dtype=np.int64)
        for b, row in enumerate(t):
            count = 0
            for p, token_id in enumerate(row):
                name = vocab.decode_token(int(token_id))
                if name in {"BOS", "_NEWLINE_"}: count = 0
                else:
                    count += data.extract_character_count(name); y[b, p] = count
        labs.append(y)
    return np.concatenate(toks), {s: np.concatenate(acts[s]) for s in SITES}, np.concatenate(labs)

train_tokens, train_acts, train_y = collect(N_TRAIN, TRAIN_SEED)
eval_tokens, eval_acts, eval_y = collect(N_EVAL, EVAL_SEED)
train_masks = {s: train_y.ravel() > 0 for s in SITES}
eval_masks = {s: eval_y.ravel() > 0 for s in SITES}
print("checkpoint", checkpoint["step"], "device", device)
print("train/eval examples", len(train_tokens), len(eval_tokens), "count classes", int(train_y.max()))
'''

probes = '''probe_rows, count_models = [], {}
for site in SITES:
    Xtr = train_acts[site].reshape(-1, 128)[train_masks[site]]; ytr = train_y.ravel()[train_masks[site]]
    Xev = eval_acts[site].reshape(-1, 128)[eval_masks[site]]; yev = eval_y.ravel()[eval_masks[site]]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    pred = ridge.predict(Xev); rounded = np.clip(np.rint(pred), yev.min(), yev.max()).astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")).fit(Xtr, ytr)
    exact = clf.predict(Xev)
    probe_rows.append({"site": site, "n_eval": len(yev), "R2": r2_score(yev, pred), "MAE": mean_absolute_error(yev, pred), "rounded_accuracy": accuracy_score(yev, rounded), "exact_accuracy": accuracy_score(yev, exact)})
    count_models[site] = ridge
probe_results = pd.DataFrame(probe_rows)
display(probe_results.round(4))
'''

nb.cells = [nbf.v4.new_markdown_cell("# Character-count emergence and layer-1 MLP accumulation\n\nThis notebook measures count decodability at four sites around layer 1 and performs causal token-length swaps. It uses identical train/evaluation splits for every site."), nbf.v4.new_code_cell(setup), nbf.v4.new_markdown_cell("## Matched count probes\n\nThe ridge probe measures continuous count information; the multinomial probe measures exact count-class accuracy. BOS and newline positions are excluded."), nbf.v4.new_code_cell(probes)]
out.write_text(nbf.writes(nb), encoding="utf-8")
print(out)
