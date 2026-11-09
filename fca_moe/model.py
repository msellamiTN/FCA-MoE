import torch
from torch import nn


class TinyTopKMoE(nn.Module):
    def __init__(self, d_model=32, n_experts=8, top_k=2, d_hidden=64):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("invalid top_k")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x, trace_collector=None, workload="default", window_size=64):
        scores = torch.softmax(self.router(x), dim=-1)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            token_indices, slot_indices = torch.where(indices == expert_id)
            if token_indices.numel():
                expert_output = expert(x[token_indices])
                output.index_add_(
                    0,
                    token_indices,
                    expert_output * weights[token_indices, slot_indices].unsqueeze(-1),
                )

        if trace_collector is not None:
            # Transfer the complete routing matrix once; avoid one CUDA sync per token.
            routed_ids_cpu = indices.detach().cpu().tolist()
            for token_idx, expert_ids in enumerate(routed_ids_cpu):
                trace_collector.add(
                    "layer0",
                    expert_ids,
                    workload=workload,
                    window=token_idx // max(window_size, 1),
                )
        return output
