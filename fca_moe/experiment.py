import json,time
from pathlib import Path
import torch
from .affinity import concept_derived_affinity,pairwise_coactivation
from .context import enumerate_concepts
from .model import TinyTopKMoE
from .placement import greedy_virtual_placement
from .synthetic import make_correlated_routing_data
from .telemetry import RoutingTrace
def run_synthetic_experiment(n_samples=4096,n_experts=8,d_model=32,epochs=30,seed=17,concept_budget=None,window_size=64,activation_rate_threshold=.1,device=None):
    torch.manual_seed(seed);device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x,targets,_=make_correlated_routing_data(n_samples,d_model,seed);x=x.to(device);targets=targets.to(device)
    model=TinyTopKMoE(d_model,n_experts,2).to(device);opt=torch.optim.AdamW(model.router.parameters(),lr=.03);t=time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model.router(x),targets);loss.backward();opt.step()
    train_s=time.perf_counter()-t;trace=RoutingTrace();model.eval()
    if device.type=="cuda":torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.no_grad():model(x,trace_collector=trace,workload="synthetic",window_size=window_size)
    if device.type=="cuda":torch.cuda.synchronize()
    infer_s=time.perf_counter()-t;token=trace.to_token_context("layer0",[f"E{i}" for i in range(n_experts)])
    window=trace.to_window_context("layer0",activation_rate_threshold,[f"E{i}" for i in range(n_experts)])
    t=time.perf_counter();concepts=enumerate_concepts(window,max_concepts=concept_budget);fca_s=time.perf_counter()-t
    fca=concept_derived_affinity(window,concepts);pair=pairwise_coactivation(window)
    place=greedy_virtual_placement(fca,{f"E{i}":1. for i in range(n_experts)},[n_experts/2]*2,[[0.,1.],[1.,0.]])
    actual=model.router(x).topk(2,-1).indices.sort(-1).values;expected=targets.topk(2,-1).indices.sort(-1).values
    return {"device":str(device),"gpu_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"torch_version":torch.__version__,"n_samples":n_samples,"n_experts":n_experts,"top_k":2,"training_seconds":train_s,"inference_seconds":infer_s,"fca_seconds":fca_s,"route_exact_match":float((actual==expected).all(-1).float().mean().item()),"token_context_objects":len(token.objects),"window_context_objects":len(window.objects),"formal_context_objects":len(window.objects),"formal_context_attributes":len(window.attributes),"concepts_returned":len(concepts.concepts),"concept_enumeration_complete":concepts.complete,"concept_candidates_examined":concepts.candidates_examined,"fca_affinity":{str(k):v for k,v in fca.items()},"pairwise_affinity":{str(k):v for k,v in pair.items()},"virtual_placement":place,"virtual_placement_objective":place["objective"],"virtual_placement_is_simulated":True}
def save_json(result,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2),encoding="utf-8")
