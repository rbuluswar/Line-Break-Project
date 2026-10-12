from pathlib import Path
import json

p = Path(__file__).with_name("08_mlp_contribution.ipynb")
nb = json.loads(p.read_text())
nb["cells"][0]["source"] = [
    "# Layer-1 branch ablations\n",
    "\n",
    "Compare the post-layer-1 trajectory after zeroing the MLP branch and after zeroing the attention branch. The interactive Plotly figures below use shared PCA coordinates and can be rotated.\n",
]
source = '''import plotly.graph_objects as go

def interactive_trajectory(trajectories, labels, title):
    coords = shared_pca(trajectories)
    fig = go.Figure()
    colors = ["royalblue", "darkorange"]
    for xyz, label, color in zip(coords, labels, colors):
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines+markers", name=label,
            line={"width": 5, "color": color},
            marker={"size": 3, "color": color},
            customdata=np.arange(len(xyz))[:, None],
            hovertemplate="position=%{customdata[0]}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(title=title, scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"}, height=700)
    return fig

# MLP-zeroed condition, using the notebook's original seed 6001.
POST, MLP = "blocks.1.hook_resid_post", "blocks.1.hook_mlp_out"
_, c = model.run_with_cache(batch.tokens, prepend_bos=False, names_filter=[POST])
clean_mlp = c[POST][0].detach().cpu().numpy()
captured = {}
def capture_mlp(x, hook): captured["post"] = x.detach().cpu(); return x
model.run_with_hooks(batch.tokens, prepend_bos=False, fwd_hooks=[(MLP, lambda x, hook: torch.zeros_like(x)), (POST, capture_mlp)])
mlp_zeroed = captured["post"][0].numpy()
display(interactive_trajectory([clean_mlp, mlp_zeroed], ["clean", "MLP zeroed"], "Interactive post-layer-1 trajectory: MLP zeroed"))

# Attention-zeroed condition, using the dedicated attention experiment's seed 6010.
attention_ctx = load_context(6010)
attention_model, attention_batch = attention_ctx.model, attention_ctx.batch
_, c = attention_model.run_with_cache(attention_batch.tokens, prepend_bos=False, names_filter=[POST])
clean_attention = c[POST][0].detach().cpu().numpy()
captured = {}
def capture_attention(x, hook): captured["post"] = x.detach().cpu(); return x
attention_model.run_with_hooks(
    attention_batch.tokens, prepend_bos=False,
    fwd_hooks=[("blocks.1.hook_attn_out", lambda x, hook: torch.zeros_like(x)), (POST, capture_attention)],
)
attention_zeroed = captured["post"][0].numpy()
display(interactive_trajectory([clean_attention, attention_zeroed], ["clean", "attention output zeroed"], "Interactive post-layer-1 trajectory: attention zeroed"))'''
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]})
p.write_text(json.dumps(nb, indent=1) + "\n")
