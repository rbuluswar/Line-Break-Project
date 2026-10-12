from pathlib import Path
import nbformat as nbf

out = Path(__file__).with_name("10_attention_zeroed_trajectory.ipynb")
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Attention-zeroed post-layer-1 trajectory\n\n"
        "Zero the layer-1 attention output while retaining the pre-layer-1 residual and the layer-1 MLP. "
        "This isolates the residual-plus-MLP condition from the clean post-layer-1 trajectory."
    ),
    nbf.v4.new_code_cell('''import sys\nfrom pathlib import Path\nPROJECT_ROOT = next(
    candidate for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd().parent.parent.parent]
    if (candidate / "results/line_breaks_results").exists()
)
sys.path.insert(0, str(PROJECT_ROOT / "notebooks/06_layer1_mechanisms"))\nfrom common import *\nEXAMPLE_SEED=6010\nctx=load_context(EXAMPLE_SEED)\nmodel,vocab,batch,cfg,device=ctx.model,ctx.vocab,ctx.batch,ctx.cfg,ctx.device\ntokens=batch.tokens[0].detach().cpu().numpy()\nprint("checkpoint",ctx.checkpoint["step"],"seed",EXAMPLE_SEED,"line width",int(batch.line_lengths[0]))'''),
    nbf.v4.new_code_cell('''ATTN_OUT,POST="blocks.1.hook_attn_out","blocks.1.hook_resid_post"\n_,c=model.run_with_cache(batch.tokens,prepend_bos=False,names_filter=[POST])\nclean=c[POST][0].detach().cpu().numpy()\ncaptured={}\ndef zero_attention(x,hook): return torch.zeros_like(x)\ndef capture(x,hook): captured["post"]=x.detach().cpu(); return x\nmodel.run_with_hooks(batch.tokens,prepend_bos=False,fwd_hooks=[(ATTN_OUT,zero_attention),(POST,capture)])\nattention_zeroed=captured["post"][0].numpy()\nprint("post-layer-1 MSE",float(np.mean((attention_zeroed-clean)**2)))\ncoords=shared_pca([clean,attention_zeroed])\nplot_trajectory(coords,["clean post layer 1","attention output zeroed"],"Post-layer-1 trajectory with attention removed","figures/layer1_attention_zeroed.png")'''),
]
out.write_text(nbf.writes(nb), encoding="utf-8")
print(out)
