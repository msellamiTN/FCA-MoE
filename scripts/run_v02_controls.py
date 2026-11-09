import argparse, json
from pathlib import Path
from fca_moe.v02_experiment import run_controlled_case, run_sensitivity_grid, save_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=int,default=4096)
    p.add_argument("--epochs",type=int,default=120)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--sensitivity",action="store_true")
    p.add_argument("--output",type=Path,default=Path("results/fca_moe_v02_controls.json"))
    a=p.parse_args()
    out={
      "positive_control":run_controlled_case(mode="window_correlated",n_samples=a.samples,window_size=64,activation_rate_threshold=.10,epochs=a.epochs,seed=a.seed),
      "negative_iid_control":run_controlled_case(mode="iid",n_samples=a.samples,window_size=64,activation_rate_threshold=.10,epochs=a.epochs,seed=a.seed)
    }
    if a.sensitivity: out["sensitivity"]=run_sensitivity_grid(n_samples=a.samples,epochs=a.epochs,seed=a.seed)
    save_json(out,a.output)
    print(json.dumps(out,indent=2))
    print("Saved:",a.output)
if __name__=="__main__": main()
