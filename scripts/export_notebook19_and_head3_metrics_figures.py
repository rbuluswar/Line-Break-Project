import json
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/03_final_attention_geometry/19_single_example_final_head_qk_geometry.ipynb"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
nb = json.loads(NB.read_text())

def plotly_fig(cell_index, output_index):
    output = nb["cells"][cell_index]["outputs"][output_index]
    return output["data"]["application/vnd.plotly.v1+json"]

def trace_arrays(fig, index):
    trace = fig["data"][index]
    def decode(value):
        if isinstance(value, dict) and "bdata" in value:
            dtype = np.dtype(value.get("dtype", "f8"))
            return np.frombuffer(base64.b64decode(value["bdata"]), dtype=dtype)
        return np.asarray(value, dtype=float)
    return decode(trace.get("x", [])), decode(trace.get("y", [])), decode(trace.get("z", []))

# Joint Q/K geometry from Notebook 19's global Q/K PCA.
fig = plotly_fig(6, 2)
q_x, q_y, q_z = trace_arrays(fig, 2)
k_x, k_y, k_z = trace_arrays(fig, 4)
fig_m = plt.figure(figsize=(8.2, 6.6))
ax = fig_m.add_subplot(111, projection="3d")
ax.plot(q_x, q_y, q_z, color="royalblue", linewidth=1.5, alpha=0.7)
ax.scatter(q_x, q_y, q_z, color="royalblue", s=18, label="Q embeddings")
ax.plot(k_x, k_y, k_z, color="crimson", linewidth=1.5, alpha=0.7)
ax.scatter(k_x, k_y, k_z, color="crimson", s=18, label="K embeddings")
ax.scatter([0], [0], [0], color="black", s=35, label="Origin")
ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3")
ax.set_title("Notebook 19: Head 3 joint Q/K geometry\n(global 3D PCA; example 78)")
ax.legend(loc="upper left")
fig_m.tight_layout()
fig_m.savefig(FIG / "notebook19_head3_joint_qk_geometry.png", dpi=220, bbox_inches="tight")
plt.close(fig_m)

# Exact normalized causal attention from the cached pattern.
fig = plotly_fig(6, 5)
heat_data = fig["data"][0]["z"]
if isinstance(heat_data, dict) and "bdata" in heat_data:
    heat = np.frombuffer(base64.b64decode(heat_data["bdata"]), dtype=np.dtype(heat_data.get("dtype", "f8"))).reshape((150, 150))
else:
    heat = np.asarray(heat_data, dtype=float)
fig_m, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(heat, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=np.nanmax(heat))
ax.set(xlabel="Key/source position", ylabel="Query position", title="Notebook 19: Head 3 normalized causal attention")
fig_m.colorbar(im, ax=ax, label="Attention probability")
fig_m.tight_layout()
fig_m.savefig(FIG / "notebook19_head3_normalized_causal_attention.png", dpi=220, bbox_inches="tight")
plt.close(fig_m)

# Query 62 actual versus global-PCA reconstructed attention.
fig = plotly_fig(9, 23)
def decode_plotly(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value.get("dtype", "f8")))
    return np.asarray(value, dtype=float)
actual = decode_plotly(fig["data"][0]["y"])
reconstructed = decode_plotly(fig["data"][1]["y"])
positions = decode_plotly(fig["data"][0]["x"]).astype(int)
fig_m, ax = plt.subplots(figsize=(9.0, 4.3))
width = 0.42
ax.bar(positions - width / 2, actual, width, color="crimson", label="Actual")
ax.bar(positions + width / 2, reconstructed, width, color="royalblue", label="3D-PCA reconstructed")
ax.set(xlabel="Previous key position", ylabel="Attention weight", title="Notebook 19: query position 62 attention")
ax.legend()
fig_m.tight_layout()
fig_m.savefig(FIG / "notebook19_query62_attention_comparison.png", dpi=220, bbox_inches="tight")
plt.close(fig_m)

# Updated overall newline precision/recall intervention chart from Notebook 29.
conditions = ["Baseline", "Mean values", "Random shuffle", "Zero values", "Previous NL", "Uniform NL", "Remove NL"]
precision = np.array([0.521989, 0.475928, 0.470898, 0.494193, 0.493568, 0.522856, 0.516489])
recall = np.array([0.999455, 0.964197, 0.952808, 0.917671, 0.999455, 0.920882, 0.973466])
x = np.arange(len(conditions))
fig_m, ax = plt.subplots(figsize=(10.2, 4.8))
ax.bar(x - 0.19, precision, 0.38, label="Newline precision", color="darkorange")
ax.bar(x + 0.19, recall, 0.38, label="Newline recall", color="seagreen")
ax.set_xticks(x, conditions, rotation=25, ha="right")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Metric")
ax.set_title("Head 3 newline metrics under value and attention interventions")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig_m.tight_layout()
fig_m.savefig(FIG / "layer2_head3_newline_intervention_metrics.png", dpi=220, bbox_inches="tight")
plt.close(fig_m)
