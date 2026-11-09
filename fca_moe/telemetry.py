from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,field
from typing import Iterable
from .context import FormalContext
@dataclass
class RoutingEvent:
    layer_id:str
    expert_ids:tuple[int,...]
    workload:str="default"
    window_id:int=0
@dataclass
class RoutingTrace:
    events:list[RoutingEvent]=field(default_factory=list)
    def add(self,layer_id:str,expert_ids:Iterable[int],workload="default",window=0):
        self.events.append(RoutingEvent(layer_id,tuple(sorted(set(map(int,expert_ids)))),workload,int(window)))
    def to_token_context(self,layer_id,attributes=None):
        selected=[e for e in self.events if e.layer_id==layer_id]
        if attributes is None:
            mx=max((max(e.expert_ids) for e in selected if e.expert_ids),default=-1); attributes=[f"E{i}" for i in range(mx+1)]
        return FormalContext.from_transactions([{f"E{i}" for i in e.expert_ids} for e in selected],attributes,f"{layer_id}:token:")
    def to_window_context(self,layer_id,activation_rate_threshold=.1,attributes=None):
        if not 0<=activation_rate_threshold<=1:raise ValueError("threshold must be in [0,1]")
        selected=[e for e in self.events if e.layer_id==layer_id]
        if attributes is None:
            mx=max((max(e.expert_ids) for e in selected if e.expert_ids),default=-1); attributes=[f"E{i}" for i in range(mx+1)]
        windows=defaultdict(list)
        for e in selected:windows[(e.workload,e.window_id)].append(e)
        tx=[]
        for events in windows.values():
            counts=defaultdict(int)
            for e in events:
                for x in e.expert_ids:counts[x]+=1
            n=len(events); tx.append({f"E{i}" for i,c in counts.items() if n and c/n>=activation_rate_threshold})
        return FormalContext.from_transactions(tx,attributes,f"{layer_id}:window:")
