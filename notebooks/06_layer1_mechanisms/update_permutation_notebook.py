from pathlib import Path
import json

p = Path(__file__).with_name("09_permute_pre_layer1_trajectory.ipynb")
nb = json.loads(p.read_text())
nb["cells"][0]["source"] = [
    "# Permute pre-layer-1 activations\n",
    "\n",
    "Compare clean, unrestricted within-segment permutation, and length-matched permutation. `PERMUTE_MODE` toggles between the middle segment and all segments.\n",
]
code = '''PERMUTE_MODE="middle"  # change to "all"
MIDDLE_SEGMENT_INDEX=1
PRE,POST="blocks.1.hook_resid_pre","blocks.1.hook_resid_post"
_,c=model.run_with_cache(batch.tokens,prepend_bos=False,names_filter=[PRE,POST]); clean_pre=c[PRE][0].detach().clone(); clean_post=c[POST][0].detach().cpu().numpy()
segments=line_segments(tokens,vocab); selected=set(range(len(segments))) if PERMUTE_MODE=="all" else {MIDDLE_SEGMENT_INDEX}; rng=np.random.default_rng(6009)
unrestricted=clean_pre.clone(); length_matched=clean_pre.clone()
lengths=np.array([token_length(t,vocab) for t in tokens])
for i,(start,end) in enumerate(segments):
 if i in selected and end-start>1:
  order=torch.as_tensor(rng.permutation(end-start),device=unrestricted.device); unrestricted[start:end]=clean_pre[start:end][order]
  for length in sorted(set(lengths[start:end])):
   if np.isnan(length): continue
   positions=np.flatnonzero(lengths[start:end]==length)+start
   if len(positions)>1:
    shuffled=positions[rng.permutation(len(positions))]
    length_matched[positions]=clean_pre[torch.as_tensor(shuffled,device=clean_pre.device)]
def run_post(replacement):
 capd={}
 def replace(x,hook): return replacement.unsqueeze(0).to(device=x.device,dtype=x.dtype)
 def cap(x,hook): capd["post"]=x.detach().cpu(); return x
 model.run_with_hooks(batch.tokens,prepend_bos=False,fwd_hooks=[(PRE,replace),(POST,cap)])
 return capd["post"][0].numpy()
unrestricted_post=run_post(unrestricted); length_matched_post=run_post(length_matched)
print("mode",PERMUTE_MODE,"selected",sorted(selected),"of",len(segments))
print("unrestricted pre MSE",float(torch.mean((unrestricted-clean_pre)**2)))
print("length-matched pre MSE",float(torch.mean((length_matched-clean_pre)**2)))
print("unrestricted post MSE",float(np.mean((unrestricted_post-clean_post)**2)))
print("length-matched post MSE",float(np.mean((length_matched_post-clean_post)**2)))
coords=shared_pca([clean_post,unrestricted_post,length_matched_post])
plot_trajectory(coords,["clean post layer 1",f"unrestricted permutation ({PERMUTE_MODE})",f"length-matched permutation ({PERMUTE_MODE})"],"Post-layer-1 trajectories after pre-layer-1 permutation","figures/layer1_pre_activation_permutation_comparison.png")'''
nb["cells"][2]["source"] = [line + "\n" for line in code.splitlines()]
p.write_text(json.dumps(nb, indent=1) + "\n")
