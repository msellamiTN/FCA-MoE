from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
@dataclass(frozen=True)
class FormalContext:
    objects: tuple[str,...]
    attributes: tuple[str,...]
    incidence: tuple[frozenset[str],...]
    @classmethod
    def from_transactions(cls, transactions: Sequence[Iterable[str]], attributes: Sequence[str], prefix: str="g"):
        a=tuple(attributes); allowed=set(a); rows=tuple(frozenset(x)&allowed for x in transactions)
        return cls(tuple(f"{prefix}{i}" for i in range(len(rows))),a,rows)
    def extent(self,intent:Iterable[str])->frozenset[int]:
        wanted=set(intent); return frozenset(i for i,row in enumerate(self.incidence) if wanted<=row)
    def intent(self,extent:Iterable[int])->frozenset[str]:
        ix=tuple(extent)
        if not ix:return frozenset(self.attributes)
        common=set(self.attributes)
        for i in ix:common.intersection_update(self.incidence[i])
        return frozenset(common)
    def closure(self,intent:Iterable[str])->frozenset[str]:return self.intent(self.extent(intent))
@dataclass(frozen=True)
class FormalConcept:
    extent:frozenset[int]
    intent:frozenset[str]
@dataclass(frozen=True)
class ConceptEnumeration:
    concepts:tuple[FormalConcept,...]
    complete:bool
    candidates_examined:int
    stop_reason:str
def _next_closure(context,current):
    attrs=context.attributes; cur=set(current)
    for i in range(len(attrs)-1,-1,-1):
        m=attrs[i]
        if m in cur:continue
        closed=context.closure(frozenset((cur&set(attrs[:i]))|{m}))
        if all((attrs[j] in closed)==(attrs[j] in cur) for j in range(i)):return closed
    return None
def enumerate_concepts(context,min_support=0.,min_intent_size=0,max_concepts=None):
    if not 0<=min_support<=1:raise ValueError("min_support must be in [0,1]")
    if min_intent_size<0:raise ValueError("min_intent_size must be non-negative")
    if max_concepts is not None and max_concepts<1:raise ValueError("max_concepts must be positive")
    current=context.closure(frozenset()); kept=[]; examined=0; n=len(context.objects)
    while current is not None:
        examined+=1; extent=context.extent(current); support=len(extent)/n if n else 0.
        if support>=min_support and len(current)>=min_intent_size:
            kept.append(FormalConcept(extent,current))
            if max_concepts is not None and len(kept)>=max_concepts:
                if _next_closure(context,current) is not None:return ConceptEnumeration(tuple(kept),False,examined,"truncated: concept budget reached")
        current=_next_closure(context,current)
    return ConceptEnumeration(tuple(kept),True,examined,"complete")
