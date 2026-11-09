from collections import defaultdict
from itertools import combinations
from .context import ConceptEnumeration,FormalContext
def pairwise_coactivation(context:FormalContext):
    counts=defaultdict(int); n=len(context.incidence)
    for row in context.incidence:
        for pair in combinations(sorted(row),2):counts[pair]+=1
    return {p:c/max(n,1) for p,c in counts.items()}
def concept_derived_affinity(context:FormalContext,enumeration:ConceptEnumeration):
    weights=defaultdict(float); n=max(len(context.objects),1)
    for c in enumeration.concepts:
        intent=sorted(c.intent)
        if len(intent)<2:continue
        contribution=(len(c.extent)/n)/(len(intent)-1)
        for p in combinations(intent,2):weights[p]+=contribution
    return dict(weights)
