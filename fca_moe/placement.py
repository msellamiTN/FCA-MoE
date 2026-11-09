def greedy_virtual_placement(affinity,expert_sizes,device_capacities,communication_cost):
    if not device_capacities:raise ValueError("at least one virtual device is required")
    n=len(device_capacities)
    if len(communication_cost)!=n or any(len(r)!=n for r in communication_cost):raise ValueError("communication cost must be square")
    placement={};used=[0.]*n;ordered=sorted(expert_sizes,key=lambda e:-sum(w for p,w in affinity.items() if e in p))
    for e in ordered:
        size=expert_sizes[e]; candidates=[d for d in range(n) if used[d]+size<=device_capacities[d]]
        if not candidates:raise ValueError(f"no virtual capacity for {e}")
        def score(d):return sum(affinity.get(tuple(sorted((e,o))),0.)*communication_cost[d][od] for o,od in placement.items())
        d=min(candidates,key=score);placement[e]=d;used[d]+=size
    obj=sum(w*communication_cost[placement[a]][placement[b]] for (a,b),w in affinity.items() if a in placement and b in placement)
    return {"placement":placement,"used_capacity":used,"objective":obj,"is_simulated":True}
