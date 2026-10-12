from pathlib import Path
import nbformat as nbf

out = Path(__file__).with_name("11_pre_mlp_line_geometry_and_spacing.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

setup = '''import sys
from pathlib import Path
PROJECT_ROOT = next(
    candidate for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]
    if (candidate / "results/line_breaks_results").exists()
)
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/06_layer1_mechanisms"))
from common import *
from sklearn.linear_model import LinearRegression

N_EXAMPLES = 128
BATCH_SIZE = 32
SEED = 6111
SITE = "blocks.1.hook_resid_mid"  # after layer-1 attention, before layer-1 MLP
ctx = load_context(SEED)
model, vocab, cfg, device = ctx.model, ctx.vocab, ctx.cfg, ctx.device
print("checkpoint", ctx.checkpoint["step"], "site", SITE)

all_tokens, all_activations = [], []
for start in range(0, N_EXAMPLES, BATCH_SIZE):
    size = min(BATCH_SIZE, N_EXAMPLES - start)
    b = data.make_batch(batch_size=size, vocab=vocab, task_cfg=cfg["task"], device=device, seed=SEED + start)
    with torch.no_grad():
        _, cache = model.run_with_cache(b.tokens, prepend_bos=False, names_filter=[SITE])
    all_tokens.append(b.tokens.detach().cpu().numpy())
    all_activations.append(cache[SITE].detach().cpu().numpy())
tokens_batch = np.concatenate(all_tokens, axis=0)
activations = np.concatenate(all_activations, axis=0)
print("examples", len(tokens_batch), "activation shape", activations.shape)

def token_len(token_id):
    name = vocab.decode_token(int(token_id))
    if name in {"BOS", "_NEWLINE_"}:
        return np.nan
    return float(data.extract_character_count(name))

def ordinary_segment(tokens, start, end):
    positions = list(range(start, end))
    return [p for p in positions if vocab.decode_token(int(tokens[p])) not in {"BOS", "_NEWLINE_"}]

def segment_records(tokens, acts):
    nls = newline_positions(tokens, vocab)
    bounds = [-1] + nls + [len(tokens)]
    records = []
    for j in range(len(bounds) - 1):
        start, end = bounds[j] + 1, bounds[j + 1]
        pos = ordinary_segment(tokens, start, end)
        if len(pos) >= 4:
            records.append({"segment": j, "positions": np.asarray(pos), "h": acts[pos], "lengths": np.asarray([token_len(tokens[p]) for p in pos])})
    return records

records = [r for tokens, acts in zip(tokens_batch, activations) for r in segment_records(tokens, acts)]
print("usable segments", len(records), "median segment length", float(np.median([len(r["positions"]) for r in records])))
'''

analysis = '''line_rows = []
spacing_rows = []
joint_rows = []
rng = np.random.default_rng(6112)

for r in records:
    h, lengths = r["h"], r["lengths"]
    n = len(h)
    mu = h.mean(axis=0)
    pca = PCA(n_components=min(3, n), random_state=6113).fit(h)
    pca1 = PCA(n_components=1, random_state=6113).fit(h)
    recon = pca.inverse_transform(pca.transform(h))
    total_ss = np.sum((h - mu) ** 2)
    resid1 = np.sum((h - pca1.inverse_transform(pca1.transform(h))) ** 2)
    r2_1 = 1 - resid1 / total_ss if total_ss > 0 else np.nan

    # Held-out line fit: fit on alternating positions and measure orthogonal
    # reconstruction error on the held-out positions.
    train = np.arange(n) % 2 == 0
    test = ~train
    if train.sum() >= 2 and test.sum() >= 2:
        pca_train = PCA(n_components=1, random_state=6114).fit(h[train])
        heldout_pred = pca_train.inverse_transform(pca_train.transform(h[test]))
        heldout_d2 = np.sum((h[test] - heldout_pred) ** 2, axis=1)
        heldout_baseline = np.sum((h[test] - h[train].mean(axis=0)) ** 2, axis=1)
        heldout_r2 = 1 - heldout_d2.sum() / heldout_baseline.sum() if heldout_baseline.sum() > 0 else np.nan
    else:
        heldout_r2 = np.nan

    coord = pca1.transform(h)[:, 0]
    cumlen = np.r_[0, np.cumsum(lengths[:-1])]
    if np.corrcoef(coord, cumlen)[0, 1] < 0:
        coord = -coord
    delta = np.diff(coord)
    euclidean_distance = np.linalg.norm(np.diff(h, axis=0), axis=1)
    x = lengths[:-1]
    valid = np.isfinite(x) & np.isfinite(delta)
    if valid.sum() >= 2:
        fit = LinearRegression().fit(x[valid, None], delta[valid])
        spacing_r2 = fit.score(x[valid, None], delta[valid])
        euclidean_r2 = LinearRegression().fit(x[valid, None], euclidean_distance[valid]).score(x[valid, None], euclidean_distance[valid])
        euclidean_corr = float(np.corrcoef(x[valid], euclidean_distance[valid])[0, 1])
        no_intercept = LinearRegression(fit_intercept=False).fit(x[valid, None], delta[valid])
        slope = float(no_intercept.coef_[0])
        corr = float(np.corrcoef(x[valid], delta[valid])[0, 1])
        positive_fraction = float(np.mean(delta[valid] > 0))
    else:
        spacing_r2, euclidean_r2, euclidean_corr, slope, corr, positive_fraction = (np.nan,) * 6

    # Joint scalar model: coordinate as a function of token index and
    # cumulative character count. This tests whether a line coordinate can be
    # predicted using the proposed length-driven update rule.
    t = np.arange(n)
    X = np.column_stack([t, cumlen])
    if n >= 4:
        joint = LinearRegression().fit(X, coord)
        joint_r2 = joint.score(X, coord)
        position_only = LinearRegression().fit(t[:, None], coord).score(t[:, None], coord)
    else:
        joint_r2, position_only = np.nan, np.nan
    line_rows.append((r2_1, heldout_r2, n))
    spacing_rows.append((spacing_r2, euclidean_r2, euclidean_corr, slope, corr, positive_fraction, n))
    joint_rows.append((joint_r2, position_only, n))

line_df = pd.DataFrame(line_rows, columns=["line_r2", "heldout_line_r2", "n"])
spacing_df = pd.DataFrame(spacing_rows, columns=["signed_spacing_r2", "euclidean_spacing_r2", "euclidean_spacing_corr", "slope_no_intercept", "signed_spacing_corr", "positive_fraction", "n"])
joint_df = pd.DataFrame(joint_rows, columns=["joint_r2", "position_only_r2", "n"])
print("LINE TEST")
print(line_df[["line_r2", "heldout_line_r2"]].describe(percentiles=[.1,.5,.9]).round(3))
print("SPACING TEST")
print(spacing_df[["signed_spacing_r2", "euclidean_spacing_r2", "euclidean_spacing_corr", "slope_no_intercept", "signed_spacing_corr", "positive_fraction"]].describe(percentiles=[.1,.5,.9]).round(3))
print("JOINT TEST")
print(joint_df[["joint_r2", "position_only_r2"]].describe(percentiles=[.1,.5,.9]).round(3))
'''

controls = '''# Pooled spacing regression and a within-segment shuffled-length null.
observed_x, observed_y, null_scores = [], [], []
for r in records:
    h, lengths = r["h"], r["lengths"]
    coord = PCA(n_components=1, random_state=6115).fit_transform(h)[:, 0]
    if np.corrcoef(coord, np.r_[0, np.cumsum(lengths[:-1])])[0, 1] < 0: coord = -coord
    delta = np.diff(coord); valid = np.isfinite(lengths[:-1])
    observed_x.extend(lengths[:-1][valid]); observed_y.extend(delta[valid])
    for _ in range(10):
        shuffled = lengths[:-1].copy(); rng.shuffle(shuffled)
        null_scores.append(np.corrcoef(shuffled[valid], delta[valid])[0, 1])
pooled = LinearRegression().fit(np.asarray(observed_x)[:, None], observed_y)
print("pooled spacing slope", float(pooled.coef_[0]), "R2", float(pooled.score(np.asarray(observed_x)[:, None], observed_y)))
print("pooled spacing correlation", float(np.corrcoef(observed_x, observed_y)[0,1]))
print("shuffled-length null correlation mean/std", float(np.nanmean(null_scores)), float(np.nanstd(null_scores)))
'''

for title, code in [
    ("Dataset-level line geometry and length spacing", setup),
    ("Tests", analysis),
    ("Controls", controls),
]:
    pass
nb.cells = [
    nbf.v4.new_markdown_cell("# Pre-MLP line geometry and token-length spacing\n\nThis notebook tests whether each newline-delimited segment lies approximately on a line in `blocks.1.hook_resid_mid`, whether consecutive spacing is proportional to the earlier token length, and whether a joint line-plus-length model predicts the trajectory."),
    nbf.v4.new_code_cell(setup),
    nbf.v4.new_code_cell(analysis),
    nbf.v4.new_code_cell(controls),
]
out.write_text(nbf.writes(nb), encoding="utf-8")
print(out)
