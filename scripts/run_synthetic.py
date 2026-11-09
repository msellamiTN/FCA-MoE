import argparse,json
from pathlib import Path
from fca_moe.experiment import run_synthetic_experiment,save_json
def main():
    p=argparse.ArgumentParser();p.add_argument("--samples",type=int,default=4096);p.add_argument("--experts",type=int,default=8);p.add_argument("--dim",type=int,default=32);p.add_argument("--epochs",type=int,default=30);p.add_argument("--seed",type=int,default=17);p.add_argument("--concept-budget",type=int,default=None);p.add_argument("--output",type=Path,default=Path("results/synthetic_run.json"));a=p.parse_args()
    r=run_synthetic_experiment(n_samples=a.samples,n_experts=a.experts,d_model=a.dim,epochs=a.epochs,seed=a.seed,concept_budget=a.concept_budget);save_json(r,a.output);print(json.dumps(r,indent=2));print(f"Saved: {a.output}")
if __name__=="__main__":main()
