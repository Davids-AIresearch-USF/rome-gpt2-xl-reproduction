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

def prompt_successes(scores, group):
    nll = np.array([[s["target_new"], s["target_true"]] for s in scores])
    assert nll.ndim == 2 and len(nll) and np.isfinite(nll).all()
    return nll[:, 1] < nll[:, 0] if group == "neighborhood" else nll[:, 0] < nll[:, 1]

def zero_prefix_results(cases, method, original_neighbors):
    wins = np.empty((len(cases), len(GROUPS)), dtype=int)
    totals = np.empty_like(wins)
    counts = {}
    still_preferred = 0
    for column, group in enumerate(GROUPS):
        for row, case in enumerate(cases):
            successes = prompt_successes(case[method]["0"][group], group)
            wins[row, column] = successes.sum()
            totals[row, column] = successes.size
            if group == "neighborhood":
                assert successes.shape == original_neighbors[row].shape
                still_preferred += np.count_nonzero(successes & original_neighbors[row])
        counts[group] = {"successful": int(wins[:, column].sum()), "total": int(totals[:, column].sum())}
    case_ids = np.array([case["case_id"] for case in cases])
    rewrite, paraphrase, neighborhood = wins.T
    record_counts = {
        "all_paraphrases": int(np.count_nonzero(paraphrase == totals[:, 1])),
        "no_paraphrases": int(np.count_nonzero(paraphrase == 0)),
        "all_neighbors": int(np.count_nonzero(neighborhood == totals[:, 2])),
        "no_paraphrases_case_ids": case_ids[paraphrase == 0].tolist(),
        "failed_rewrite_case_ids": case_ids[rewrite == 0].tolist(),
    }
    retention = {"originally_preferred": sum(np.count_nonzero(row) for row in original_neighbors),
                 "still_preferred": still_preferred}
    return 100 * (wins / totals), {"prompts": counts, "records": record_counts, "neighborhood_retention": retention}

def prefix_results(cases, method, length):
    groups = ("paraphrase", "neighborhood")
    rates = np.empty((len(cases), len(groups)))
    transitions = {}
    for column, group in enumerate(groups):
        counts = {"lost": 0, "gained": 0, "kept": 0, "neither": 0}
        for row, case in enumerate(cases):
            prompts = case[method]
            before = prompt_successes(prompts["0"][group], group)
            after = np.stack([prompt_successes(prompts[f"{length}_{j}"][group], group) for j in (1, 2)])
            assert after.shape == (2, before.size)
            rates[row, column] = (100 * after.mean(axis=1)).mean()
            counts["lost"] += np.count_nonzero(before & ~after)
            counts["gained"] += np.count_nonzero(~before & after)
            counts["kept"] += np.count_nonzero(before & after)
            counts["neither"] += np.count_nonzero(~before & ~after)
        transitions[group] = counts
    return rates, transitions

def estimate(point, draws):
    low, high = np.quantile(draws, [.025, .975],axis=0)
    return {"mean": np.asarray(point).tolist(), "low":low.tolist(), "high":high.tolist()}

def summarize(cases):
    count=len(cases)
    draw=np.random.default_rng(42).integers(count,size=(10000,count))
    means, effects, paired = {}, {}, {}
    editing_counts = {}
    prefix_transitions = {32: {}, 128: {}}
    table = {}
    measurements = {}
    original_neighbors = [prompt_successes(case["unchanged"]["0"]["neighborhood"], "neighborhood") for case in cases]
    for method in METHODS:
        scores, editing_counts[method] = zero_prefix_results(cases, method, original_neighbors)
        table[method]=scores
        avg=scores.mean(axis=0); samples=scores[draw].mean(axis=1)
        means[method]=estimate(np.r_[avg,harmonic(avg)],np.c_[samples,harmonic(samples)])
        if method in ("unchanged","ROME"):
            for n in (32,128):
                pref, prefix_transitions[n][method] = prefix_results(cases, method, n)
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
    return {"n":count,"scores":means,"editing_counts":editing_counts,"prefix":effects,"rome_minus_comparator":paired,
            "prefix_transitions":prefix_transitions, "paper_table_4":PAPER,"seconds_total":sum(c["seconds"] for c in cases),
            "max_gpu_gb":max(c["peak_gb"] for c in cases)}

def traces(z, plan):
    selected = plan["selected"]
    fields=["first_subject","middle_subject","last_subject","first_after","middle_after","last_prompt"]
    summary={}
    baselines = {}
    middle_subject_mlp = []
    example = {"known_id": selected[0], "restorations": {}}
    for kind, focal_group in (("mlp", "last_subject"), ("attn", "last_prompt")):
        groups={name:[] for name in fields}
        for record in selected:
            suffix=f"trace_{record}_{kind}.npz"
            with np.load(io.BytesIO(z.read("rome_outputs/"+suffix)),allow_pickle=False) as data:
                assert data["correct_prediction"] and data["window"]==10
                restored = data["scores"]
                scores = restored - data["low_score"]
                first,after=map(int,data["subject_range"]); end=len(scores)-1
                baseline = (float(data["high_score"]), float(data["low_score"]))
                if kind == "mlp":
                    baselines[record] = baseline
                    if after - first > 2:
                        middle_subject_mlp.append(restored[first + 1:after - 1])
                else:
                    np.testing.assert_allclose(baseline, baselines[record], rtol=1e-6, atol=1e-8)
                if record == selected[0]:
                    if kind == "mlp":
                        example.update(prompt="".join(data["input_tokens"].tolist()), answer_token=data["answer"].item(),
                                       clean_probability=baseline[0], corrupted_probability=baseline[1])
                    position = after - 1 if focal_group == "last_subject" else end
                    example_curve = restored[position]
                slices=[scores[first:first+1],scores[first+1:after-1],scores[after-1:after],
                        scores[after:after+1],scores[after+1:end],scores[end:end+1]]
                for name,arr in zip(fields,slices):
                    if len(arr):
                        groups[name].append(arr.mean(0))
        summary[kind]={name:{"n":len(v),"effect":np.mean(v,0).tolist()} for name,v in groups.items() if v}
        effects = summary[kind][focal_group]["effect"]
        peak_layer = int(np.argmax(effects))
        example["restorations"][kind] = {
            "token_group": focal_group, "sample_peak_center_layer": peak_layer,
            "sample_mean_probability_gain": effects[peak_layer],
            "restored_probability": float(example_curve[peak_layer]),
        }
    clean, corrupted = np.mean(list(baselines.values()), axis=0).tolist()
    details = {"eligible_count": len(plan["eligible"]), "selected_count": len(selected),
               "mean_clean_probability": clean, "mean_corrupted_probability": corrupted, "first_statement": example}
    if middle_subject_mlp:
        statement_means = summary["mlp"]["middle_subject"]["effect"]
        layer = int(np.argmax(statement_means))
        middle_tokens = np.concatenate(middle_subject_mlp)
        details["middle_subject_mlp_comparison"] = {
            "statement_weighted_peak_center_layer": layer, "statements": len(middle_subject_mlp),
            "token_positions": len(middle_tokens), "statement_weighted_probability_gain": statement_means[layer],
            "pooled_restored_minus_overall_corrupted": float(middle_tokens[:, layer].mean()) - corrupted,
        }
    return summary, details

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
            data["traces"], data["trace_details"] = traces(z, trace_plan)
    with args.output.open("x",encoding="utf8") as f:json.dump(data,f,indent=2,allow_nan=False)

if __name__=="__main__":
    main()
