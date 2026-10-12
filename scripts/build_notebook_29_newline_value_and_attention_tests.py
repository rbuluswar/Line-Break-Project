from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
out = root / "notebooks/03_final_attention_geometry/29_head3_newline_values_and_attention_causality.ipynb"
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Head 3 newline values and attention causality

This notebook tests whether Head 3's newline source tokens carry interchangeable value information, and whether attention to newline tokens predicts or causes newline outputs. It uses value interventions, attention-pattern redistribution, and attention removal on a fresh evaluation batch."""))

cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = next(c for c in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent] if (c / 'results/line_breaks_results').exists())
sys.path.insert(0, str(PROJECT_ROOT / 'src')) if str(PROJECT_ROOT / 'src') not in sys.path else None
import project_main.checkpoints as project_checkpoints
import project_main.data as project_data
import project_main.model as project_model
import project_main.tokens as project_tokens

RUN_DIR = PROJECT_ROOT / 'results/line_breaks_results'
with open(RUN_DIR / 'config.json') as f: cfg = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab = project_tokens.build_vocab(cfg['task'])
model = project_model.build_model(cfg=cfg, device=device)
checkpoint = project_checkpoints.load_checkpoint(path=RUN_DIR / 'checkpoints/final.pt', model=model, map_location=device)
model.eval()
HEAD = 3
FINAL_LAYER = cfg['model']['n_layers'] - 1
PATTERN_SITE = f'blocks.{FINAL_LAYER}.attn.hook_pattern'
VALUE_SITE = f'blocks.{FINAL_LAYER}.attn.hook_v'
BATCH_SIZE = 2048
print(f'Loaded checkpoint step {checkpoint["step"]} on {device}; final layer={FINAL_LAYER}, head={HEAD}')"""))

cells.append(nbf.v4.new_code_cell("""batch = project_data.make_batch(batch_size=BATCH_SIZE, vocab=vocab, task_cfg=cfg['task'], device=device, seed=29001)
tokens, targets = batch.tokens, batch.targets
with torch.no_grad():
    baseline_logits, cache = model.run_with_cache(tokens, prepend_bos=False, names_filter=lambda n: n in {PATTERN_SITE, VALUE_SITE})
pattern = cache[PATTERN_SITE].detach()
values = cache[VALUE_SITE].detach()
W_O_HEAD = model.W_O[FINAL_LAYER, HEAD].detach()
newline_id = vocab.newline_id
source_newline = tokens == newline_id
target_newline = targets == newline_id
torch.manual_seed(29003)
shuffle_permutations = []
for b in range(tokens.shape[0]):
    pos = torch.where(source_newline[b])[0]
    shuffle_permutations.append(torch.randperm(len(pos), device=pos.device) if len(pos) else None)
print(f'Batch shape={tuple(tokens.shape)}; source newline rate={source_newline.float().mean().item():.3f}; target newline rate={target_newline.float().mean().item():.3f}')"""))

cells.append(nbf.v4.new_code_cell("""def metrics(logits):
    pred = logits.argmax(-1)
    pred_nl = pred == newline_id
    other = logits.masked_fill(torch.nn.functional.one_hot(torch.full_like(pred, newline_id), logits.shape[-1]).bool(), -torch.inf).max(-1).values
    margin = logits[..., newline_id] - other
    tp = (pred_nl & target_newline).sum().item()
    return {'accuracy': (pred == targets).float().mean().item(), 'newline_precision': tp / pred_nl.sum().item(), 'newline_recall': tp / target_newline.sum().item(), 'newline_frequency': pred_nl.float().mean().item(), 'newline_margin': margin, 'predictions': pred}

base = metrics(baseline_logits)
def run(site, fn):
    with torch.no_grad(): return model.run_with_hooks(tokens, prepend_bos=False, fwd_hooks=[(site, fn)])

def value_patch(kind):
    def patch(v, hook):
        v = v.clone()
        for b in range(v.shape[0]):
            pos = torch.where(source_newline[b])[0]
            if len(pos) == 0: continue
            if kind == 'mean': v[b, pos, HEAD] = v[b, pos, HEAD].mean(0, keepdim=True)
            elif kind == 'shuffle':
                v[b, pos, HEAD] = v[b, pos[shuffle_permutations[b]], HEAD]
            elif kind == 'zero': v[b, pos, HEAD] = 0
        return v
    return patch

def pattern_patch(kind):
    def patch(p, hook):
        p = p.clone()
        for b in range(p.shape[0]):
            for q in range(p.shape[2]):
                pos = torch.where(source_newline[b, :q])[0]
                if len(pos) == 0: continue
                total = p[b, HEAD, q, pos].sum(); p[b, HEAD, q, pos] = 0
                if kind == 'uniform': p[b, HEAD, q, pos] = total / len(pos)
                elif kind == 'previous': p[b, HEAD, q, pos[-1]] = total
                elif kind == 'remove': p[b, HEAD, q] /= p[b, HEAD, q].sum()
        return p
    return patch

def summarize(condition, logits):
    m = metrics(logits)
    return {'condition': condition, **{k: v for k, v in m.items() if not isinstance(v, torch.Tensor)}, 'mean_abs_logit_change': (logits-baseline_logits).abs().mean().item(), 'prediction_flip_rate': (m['predictions'] != base['predictions']).float().mean().item(), 'mean_abs_margin_change': (m['newline_margin']-base['newline_margin']).abs().mean().item()}
print({k: v for k, v in base.items() if not isinstance(v, torch.Tensor)})"""))

cells.append(nbf.v4.new_markdown_cell("""## Do newline source values carry the same information?

Attention is held fixed. Within each sequence, newline values are replaced by their mean, permuted among newline positions, or zeroed. Small mean/shuffle effects indicate interchangeability; zeroing measures the total contribution of newline values."""))
cells.append(nbf.v4.new_code_cell("""value_table = pd.DataFrame([summarize('baseline', baseline_logits)] + [summarize(f'value_{k}', run(VALUE_SITE, value_patch(k))) for k in ('mean', 'shuffle', 'zero')])
display(value_table)

rows = []
for b in range(tokens.shape[0]):
    for q in range(tokens.shape[1]):
        pos = torch.where(source_newline[b, :q])[0]
        w = pattern[b, HEAD, q, pos] if len(pos) else torch.zeros(1, device=device)
        rows.append({'newline_attention': w.sum().item(), 'previous_newline_attention': w[-1].item() if len(pos) else 0, 'target_newline': bool(target_newline[b,q]), 'predicted_newline': bool(base['predictions'][b,q] == newline_id), 'margin': base['newline_margin'][b,q].item()})
attention_df = pd.DataFrame(rows)
for x in ('newline_attention', 'previous_newline_attention'):
    for y in ('target_newline', 'predicted_newline'):
        print(x, y, 'AUC=', roc_auc_score(attention_df[y], attention_df[x]))
display(attention_df[['newline_attention','previous_newline_attention','margin']].corr())
attention_df['bin'] = pd.qcut(attention_df.newline_attention, 10, duplicates='drop')
display(attention_df.groupby('bin', observed=True).agg(n=('margin','size'), attention=('newline_attention','mean'), target_newline=('target_newline','mean'), predicted_newline=('predicted_newline','mean'), margin=('margin','mean')))"""))

cells.append(nbf.v4.new_markdown_cell("""## Does concentration among newline keys matter?

These interventions preserve the total Head 3 attention mass on newline keys, but put it all on the previous newline or spread it uniformly across previous newlines. Removal renormalizes the remaining attention. A difference between previous and uniform means that which newline is attended to matters, not just total newline mass."""))
cells.append(nbf.v4.new_code_cell("""pattern_table = pd.DataFrame([summarize('baseline', baseline_logits)] + [summarize(f'attention_{k}', run(PATTERN_SITE, pattern_patch(k))) for k in ('previous', 'uniform', 'remove')])
display(pattern_table)

rows = []
for b in range(tokens.shape[0]):
    for q in range(tokens.shape[1]):
        pos = torch.where(source_newline[b, :q])[0]
        if len(pos) == 0: continue
        w = pattern[b, HEAD, q, pos]
        actual = (w[:,None] * values[b, pos, HEAD]).sum(0) @ W_O_HEAD
        concentrated = (w.sum() * values[b, pos[-1], HEAD]) @ W_O_HEAD
        rows.append({'newline_attention': w.sum().item(), 'difference_norm': (actual-concentrated).norm().item(), 'relative_difference': (actual-concentrated).norm().item()/(actual.norm().item()+1e-8), 'target_newline': bool(target_newline[b,q])})
display(pd.DataFrame(rows).groupby('target_newline').agg(n=('difference_norm','size'), newline_attention=('newline_attention','mean'), difference_norm=('difference_norm','mean'), relative_difference=('relative_difference','mean')))
print('Interpretation: compare value_mean/value_shuffle for interchangeability, attention_previous vs attention_uniform for concentration effects, and attention_remove for causal necessity.')"""))

cells.append(nbf.v4.new_markdown_cell("""## Newline-only conditional effects

The following table avoids conflating ordinary-token behavior with newline behavior. It reports effects among (a) positions that baseline predicts as newline and (b) positions whose true target is newline. `predicted_to_nonnewline` measures newline predictions lost under the intervention; `nonnewline_to_predicted` measures new newline predictions created. Margin changes are for the newline-vs-best-nonnewline logit margin."""))
cells.append(nbf.v4.new_code_cell("""def conditional_effects(condition, logits):
    m = metrics(logits)
    pred0 = base['predictions'] == newline_id
    pred1 = m['predictions'] == newline_id
    rows = []
    for subset, mask in [('baseline_predicted_newline', pred0), ('true_target_newline', target_newline)]:
        n = mask.sum().item()
        rows.append({
            'condition': condition, 'subset': subset, 'n': n,
            'prediction_changed': (pred1[mask] != pred0[mask]).float().mean().item(),
            'predicted_to_nonnewline': (pred0[mask] & ~pred1[mask]).float().mean().item(),
            'nonnewline_to_predicted': (~pred0[mask] & pred1[mask]).float().mean().item(),
            'newline_margin_change': (m['newline_margin'][mask] - base['newline_margin'][mask]).mean().item(),
            'newline_margin_abs_change': (m['newline_margin'][mask] - base['newline_margin'][mask]).abs().mean().item(),
        })
    return rows

conditional_rows = conditional_effects('baseline', baseline_logits)
for kind in ('mean', 'shuffle', 'zero'):
    conditional_rows += conditional_effects(f'value_{kind}', run(VALUE_SITE, value_patch(kind)))
for kind in ('previous', 'uniform', 'remove'):
    conditional_rows += conditional_effects(f'attention_{kind}', run(PATTERN_SITE, pattern_patch(kind)))
conditional_table = pd.DataFrame(conditional_rows)
display(conditional_table)
print('Interpretation: compare the two subsets within each intervention. If newline values or newline attention have the same impact in both senses, these conditional rows should be similar; differences indicate target-conditioned versus prediction-conditioned effects.')"""))

cells.append(nbf.v4.new_markdown_cell("""## Logit differences at newline-relevant positions

This repeats the intervention comparison using only positions where either the baseline prediction or the true target is `_NEWLINE_`. The main quantity is the change in the newline-vs-best-non-newline logit margin; this reveals effects even when the argmax prediction does not flip."""))
cells.append(nbf.v4.new_code_cell("""relevant = (base['predictions'] == newline_id) | target_newline
def logit_diff_row(condition, logits):
    m = metrics(logits)
    delta_margin = (m['newline_margin'] - base['newline_margin'])[relevant]
    delta_newline_logit = (logits[..., newline_id] - baseline_logits[..., newline_id])[relevant]
    return {
        'condition': condition, 'n_relevant': int(relevant.sum().item()),
        'mean_delta_newline_logit': delta_newline_logit.mean().item(),
        'mean_abs_delta_newline_logit': delta_newline_logit.abs().mean().item(),
        'mean_delta_newline_margin': delta_margin.mean().item(),
        'mean_abs_delta_newline_margin': delta_margin.abs().mean().item(),
        'median_delta_newline_margin': delta_margin.median().item(),
        'p10_delta_newline_margin': torch.quantile(delta_margin, 0.10).item(),
        'p90_delta_newline_margin': torch.quantile(delta_margin, 0.90).item(),
    }

logit_diff_rows = [logit_diff_row('baseline', baseline_logits)]
for kind in ('mean', 'shuffle', 'zero'):
    logit_diff_rows.append(logit_diff_row(f'value_{kind}', run(VALUE_SITE, value_patch(kind))))
for kind in ('previous', 'uniform', 'remove'):
    logit_diff_rows.append(logit_diff_row(f'attention_{kind}', run(PATTERN_SITE, pattern_patch(kind))))
display(pd.DataFrame(logit_diff_rows))
print('Positive margin change favors newline; negative margin change disfavors newline. The relevant set is the union of baseline-newline and true-newline positions.')"""))

cells.append(nbf.v4.new_markdown_cell("""## Logit differences restricted to true newline targets

This is the same calculation restricted to positions where the true target is `_NEWLINE_`. It measures how each intervention changes the newline logit and the newline-vs-best-non-newline margin specifically at genuine line-break decisions."""))
cells.append(nbf.v4.new_code_cell("""relevant = target_newline
target_logit_rows = [logit_diff_row('baseline', baseline_logits)]
for kind in ('mean', 'shuffle', 'zero'):
    target_logit_rows.append(logit_diff_row(f'value_{kind}', run(VALUE_SITE, value_patch(kind))))
for kind in ('previous', 'uniform', 'remove'):
    target_logit_rows.append(logit_diff_row(f'attention_{kind}', run(PATTERN_SITE, pattern_patch(kind))))
display(pd.DataFrame(target_logit_rows))
print('All rows are now computed only over true newline targets; positive margin change favors newline.')"""))

cells.append(nbf.v4.new_markdown_cell("""## Overall newline metrics

This final table reports the model-wide newline precision and recall for the baseline and every intervention. The delta columns are percentage-point changes relative to the baseline."""))
cells.append(nbf.v4.new_code_cell("""metric_rows = []
conditions = [('baseline', baseline_logits)]
conditions += [(f'value_{kind}', run(VALUE_SITE, value_patch(kind))) for kind in ('mean', 'shuffle', 'zero')]
conditions += [(f'attention_{kind}', run(PATTERN_SITE, pattern_patch(kind))) for kind in ('previous', 'uniform', 'remove')]
for condition, logits in conditions:
    m = metrics(logits)
    metric_rows.append({'condition': condition, 'newline_precision': m['newline_precision'], 'newline_recall': m['newline_recall'], 'newline_frequency': (m['predictions'] == newline_id).float().mean().item()})
metric_table = pd.DataFrame(metric_rows)
baseline_precision = metric_table.loc[metric_table.condition == 'baseline', 'newline_precision'].iloc[0]
baseline_recall = metric_table.loc[metric_table.condition == 'baseline', 'newline_recall'].iloc[0]
metric_table['precision_delta_pp'] = 100 * (metric_table['newline_precision'] - baseline_precision)
metric_table['recall_delta_pp'] = 100 * (metric_table['newline_recall'] - baseline_recall)
display(metric_table)
print('Precision and recall are fractions in the first columns; delta columns are percentage points relative to baseline.')"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}})
out.write_text(nbf.writes(nb), encoding='utf-8')
print(out)
