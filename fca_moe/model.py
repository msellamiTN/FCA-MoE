import torch
from torch import nn
class TinyTopKMoE(nn.Module):
    def __init__(self,d_model=32,n_experts=8,top_k=2,d_hidden=64):
        super().__init__()
        if not 1<=top_k<=n_experts:raise ValueError("invalid top_k")
        self.n_experts=n_experts;self.top_k=top_k;self.router=nn.Linear(d_model,n_experts)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d_model,d_hidden),nn.GELU(),nn.Linear(d_hidden,d_model)) for _ in range(n_experts)])
    def forward(self,x,trace_collector=None,workload="default",window_size=64):
        scores=torch.softmax(self.router(x),-1); weights,indices=torch.topk(scores,self.top_k,-1);weights=weights/weights.sum(-1,keepdim=True);out=torch.zeros_like(x)
        for eid,expert in enumerate(self.experts):
            ti,si=torch.where(indices==eid)
            if ti.numel():out.index_add_(0,ti,expert(x[ti])*weights[ti,si].unsqueeze(-1))
        if trace_collector is not None:
            for i in range(indices.shape[0]):trace_collector.add("layer0",indices[i].detach().cpu().tolist(),workload,i//max(window_size,1))
        return out
