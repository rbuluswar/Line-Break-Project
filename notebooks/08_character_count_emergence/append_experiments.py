import json
from pathlib import Path

p = Path(__file__).with_name("01_count_emergence_and_mlp_accumulation.ipynb")
nb = json.loads(p.read_text())

swaps = '''ids_by_length = {}
for token_id in range(vocab.size):
    name = vocab.decode_token(token_id)
    if name not in {"BOS", "_NEWLINE_"}:
        ids_by_length.setdefault(data.extract_character_count(name), []).append(token_id)

def count_at(row, p):
    count = 0
    for token_id in row[:p + 1]:
        name = vocab.decode_token(int(token_id))
        count = 0 if name == "_NEWLINE_" else count + data.extract_character_count(name)
    return count

def target_position(row):
    nls = [i for i, token_id in enumerate(row) if int(token_id) == vocab.newline_id]
    if len(nls) < 2:
        return None
    candidates = [p for p in range(nls[0] + 1, nls[1]) if int(row[p]) != vocab.newline_id]
    return candidates[len(candidates) // 2] if len(candidates) >= 5 else None

post_model = count_models["blocks.1.hook_resid_post"]
scaler = post_model.named_steps["standardscaler"]
ridge = post_model.named_steps["ridge"]
count_direction = ridge.coef_ / scaler.scale_; rng = np.random.default_rng(8014)
swap_rows = []
for sample in range(128):
    batch = data.make_batch(1, vocab, cfg["task"], device, 8013 + sample)
    clean = batch.tokens.detach().cpu().numpy()[0]
    p = target_position(clean)
    if p is None:
        continue
    old_id = int(clean[p])
    old_len = data.extract_character_count(vocab.decode_token(old_id))
    new_len = int(rng.choice([x for x in ids_by_length if x != old_len]))
    counter = clean.copy(); counter[p] = int(rng.choice(ids_by_length[new_len]))
    counter_batch = batch.tokens.clone(); counter_batch[0] = torch.as_tensor(counter, device=device)
    with torch.no_grad():
        _, c0 = model.run_with_cache(batch.tokens, prepend_bos=False, names_filter=SITES)
        _, c1 = model.run_with_cache(counter_batch, prepend_bos=False, names_filter=SITES)
    for site in SITES:
        v0 = c0[site][0, p].detach().cpu().numpy(); v1 = c1[site][0, p].detach().cpu().numpy()
        swap_rows.append({"sample": sample, "site": site, "delta_length": new_len - old_len, "delta_count": count_at(counter, p) - count_at(clean, p), "delta_coordinate": float((v1 - v0) @ count_direction)})
swap_results = pd.DataFrame(swap_rows); swap_summary = []
for site, group in swap_results.groupby("site"):
    fit = Ridge(alpha=1e-6).fit(group[["delta_length"]], group["delta_coordinate"])
    swap_summary.append({"site": site, "n_swaps": len(group), "correlation": np.corrcoef(group.delta_length, group.delta_coordinate)[0, 1], "slope": fit.coef_[0], "R2": fit.score(group[["delta_length"]], group.delta_coordinate)})
print("Swap changes projected onto the post-layer-1 count direction")
display(pd.DataFrame(swap_summary).round(4))
'''

plots = '''import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
table = probe_results.copy(); table["label"] = table.site.map(short)
axes[0].bar(table.label, table.R2); axes[0].set_title("Held-out linear count R2"); axes[0].set_ylabel("R2")
axes[1].bar(table.label, table.exact_accuracy); axes[1].set_title("Held-out exact-count accuracy"); axes[1].set_ylabel("accuracy")
for ax in axes:
    ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
plt.show()
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for site, group in swap_results.groupby("site"):
    ax.scatter(group.delta_length, group.delta_coordinate, s=12, alpha=.45, label=short[site])
lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], "k--", label="unit slope")
ax.set_xlabel("intervened token-length change"); ax.set_ylabel("change along post-layer-1 count direction")
ax.legend(frameon=False); ax.grid(alpha=.25); plt.show()
'''

nb["cells"].extend([
    {"cell_type": "markdown", "metadata": {}, "source": ["## Causal token-length swaps\n", "\n", "At a middle-segment content position, replace the current token with a token of a different length while leaving the other input tokens fixed. The immediate count changes by exactly the length difference.\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": swaps.splitlines(True)},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Figures\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": plots.splitlines(True)},
])
p.write_text(json.dumps(nb, indent=1))
print(p)
