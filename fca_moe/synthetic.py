import torch
def make_correlated_routing_data(n_samples=4096,d_model=32,seed=17):
    g=torch.Generator().manual_seed(seed); x=torch.randn(n_samples,d_model,generator=g); family=torch.randint(0,4,(n_samples,),generator=g)
    targets=torch.zeros(n_samples,8); pairs=[(0,1),(2,3),(4,5),(6,7)]
    for i,f in enumerate(family.tolist()):
        a,b=pairs[f]; targets[i,a]=1.; targets[i,b]=1.
    x[:,:4]+=torch.nn.functional.one_hot(family,4).float()*1.5
    return x,targets,family
