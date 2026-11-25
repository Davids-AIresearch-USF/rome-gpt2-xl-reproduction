"""Summarize matched CounterFact cases and causal traces from the Colab experiment."""
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np

METHODS = ("unchanged", "FT", "FT+L", "ROME")
GROUPS = ("rewrite", "paraphrase", "neighborhood")
PAPER = {"unchanged": [22.2, 24.7, 78.1, 30.5], "FT": [100., 87.9, 40.4, 65.1],
         "FT+L": [99.1, 48.7, 70.3, 66.9], "ROME": [100., 96.4, 75.4, 89.2]}

def harmonic(arr):
    with np.errstate(divide="ignore"):
        return 3 / (1 / arr).sum(axis=-1)

def outcome(scores, group):
    nll = np.array([[s["target_new"], s["target_true"]] for s in scores])
    assert nll.ndim == 2 and len(nll) and np.isfinite(nll).all()
    success = nll[:, 1] < nll[:, 0] if group == "neighborhood" else nll[:, 0] < nll[:, 1]
    return 100 * success.mean()

def estimate(point, draws):
    low, high = np.quantile(draws, [.025, .975],axis=0)
    return {"mean": np.asarray(point).tolist(), "low":low.tolist(), "high":high.tolist()}

def summarize(cases):
    count=len(cases)
    draw=np.random.default_rng(42).integers(count,size=(10000,count))
    means, effects, paired = {}, {}, {}
    table = {}
    measurements = {}
    for method in METHODS:
        scores=np.array([[outcome(c[method]["0"][g],g) for g in GROUPS] for c in cases])
        table[method]=scores
        avg=scores.mean(axis=0); samples=scores[draw].mean(axis=1)
        means[method]=estimate(np.r_[avg,harmonic(avg)],np.c_[samples,harmonic(samples)])
        if method in ("unchanged","ROME"):
            for n in (32,128):
                pref=np.array([[np.mean([outcome(c[method][f"{n}_{j}"][g],g) for j in (1,2)])
                                for g in GROUPS[1:]] for c in cases])
                measurements[(method,n)]=pref-scores[:,1:]
    for n in (32,128):
        effect=measurements["ROME",n]-measurements["unchanged",n]
        effects[n]={"paired_difference": estimate(effect.mean(0),effect[draw].mean(1))}
        for method in ("unchanged","ROME"):
            change=measurements[method,n]
            effects[n][method]=estimate(change.mean(0),change[draw].mean(1))
    for name in METHODS[:-1]:
        diff=table["ROME"]-table[name]
        paired[name]=estimate(diff.mean(0),diff[draw].mean(1))
    return {"n":count,"scores":means,"prefix":effects,"rome_minus_comparator":paired,
            "paper_table_4":PAPER,"seconds_total":sum(c["seconds"] for c in cases),
            "max_gpu_gb":max(c["peak_gb"] for c in cases)}

def traces(z, selected):
    fields=["first_subject","middle_subject","last_subject","first_after","middle_after","last_prompt"]
    summary={}
    for kind in ("mlp","attn"):
        groups={name:[] for name in fields}
        for record in selected:
            suffix=f"trace_{record}_{kind}.npz"
            with np.load(io.BytesIO(z.read("rome_outputs/"+suffix)),allow_pickle=False) as data:
                assert data["correct_prediction"] and data["window"]==10
                scores=data["scores"]-data["low_score"]
                first,after=map(int,data["subject_range"]); end=len(scores)-1
                slices=[scores[first:first+1],scores[first+1:after-1],scores[after-1:after],
                        scores[after:after+1],scores[after+1:end],scores[end:end+1]]
                for name,arr in zip(fields,slices):
                    if len(arr):
                        groups[name].append(arr.mean(0))
        summary[kind]={name:{"n":len(v),"effect":np.mean(v,0).tolist()} for name,v in groups.items() if v}
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--checkpoint",action="store_true",help="Analyze completed pilot cases only.")
    args=parser.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        plan=json.loads(z.read("rome_outputs/plan.json"))
        phase="pilot" if args.checkpoint else "eval"
        order=plan["pilot" if args.checkpoint else "evaluation"]
        assert len(set(order))==len(order) and set(plan["pilot"]).isdisjoint(plan["evaluation"])
        present=set(z.namelist())
        cases=[]
        for case_id in order:
            name=f"rome_outputs/{phase}_{case_id:05}.json"
            if args.checkpoint and name not in present:continue
            case=json.loads(z.read(name));assert case["case_id"]==case_id
            cases.append(case)
        data=summarize(cases)
        data["phase"]=phase
        if not args.checkpoint:
            trace_plan=json.loads(z.read("rome_outputs/trace_plan.json"))
            assert len(set(trace_plan["selected"]))==50 and set(trace_plan["selected"])<=set(trace_plan["eligible"])
            data["traces"]=traces(z,trace_plan["selected"])
    with args.output.open("x",encoding="utf8") as f:json.dump(data,f,indent=2,allow_nan=False)

if __name__=="__main__":
    main()
