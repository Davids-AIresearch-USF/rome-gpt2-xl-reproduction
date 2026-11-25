import io
import json
import argparse
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Plot the editing scores and causal traces.")
parser.add_argument("archive", type=Path)
parser.add_argument("summary", type=Path)
parser.add_argument("output", type=Path, help="New directory for the three figures.")
args = parser.parse_args()
archive, summary_path, output = args.archive, args.summary, args.output
output.mkdir(parents=True, exist_ok=False)
summary = json.loads(summary_path.read_text(encoding='utf-8'))
methods = ['unchanged', 'FT', 'FT+L', 'ROME']
method_labels = ['Original', 'FT', 'FT+L', 'ROME']
colors = ['#555555', '#D55E00', '#009E73', '#0072B2']
markers = ['o', 'v', 's', 'D']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.labelsize': 9, 'legend.fontsize': 8})

fig, axs = plt.subplots(1, 3, figsize=(6.7, 2.12), sharey=True, layout='constrained')
for j, (ax, title) in enumerate(zip(axs, ['Rewrite', 'Paraphrase', 'Neighborhood'])):
    for i, method in enumerate(methods):
        result = summary['scores'][method]
        center, low, high = [result[key][j] for key in ('mean', 'low', 'high')]
        ax.errorbar(i, center, yerr=[[center - low], [high - center]], fmt=markers[i], color=colors[i], capsize=3)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_xticks(range(4), method_labels, fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis='y', color='0.9', linewidth=0.5)
axs[0].set_ylabel('Success (%)')
fig.savefig(output / 'editing_scores.pdf')
plt.close(fig)

with zipfile.ZipFile(archive) as z:
    trace_plan = json.loads(z.read('rome_outputs/trace_plan.json'))
    case_ids = trace_plan['selected']
    settings = [('Last subject token', 'last_subject'), ('Last prompt token', 'last_prompt')]
    arrays = {group: {kind: [] for kind in ('mlp', 'attn')} for _, group in settings}
    for case_id in case_ids:
        for kind in ('mlp', 'attn'):
            name = f'rome_outputs/trace_{case_id}_{kind}.npz'
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as trace:
                assert trace['correct_prediction'] and trace['window'] == 10
                values = (trace['scores'] - trace['low_score']) * 100
                arrays['last_subject'][kind].append(values[trace['subject_range'][1] - 1])
                arrays['last_prompt'][kind].append(values[-1])

sample_indices = np.random.default_rng(42).integers(len(case_ids), size=(10000, len(case_ids)))
fig, axs = plt.subplots(1, 2, figsize=(6.7, 2.28), sharey=True, layout='constrained')
for ax, (title, group) in zip(axs, settings):
    ax.set_title(title, loc='left', fontweight='bold')
    ax.axhline(0, color='.6', linewidth=.65)
    for kind, label, color, pattern in [('mlp', 'MLP', '#0072B2', '-'), ('attn', 'Attention', '#D55E00', '--')]:
        values = np.asarray(arrays[group][kind], dtype=np.float64)
        avg = values.mean(0)
        np.testing.assert_allclose(avg, np.array(summary['traces'][kind][group]['effect']) * 100, rtol=1e-5, atol=1e-5)
        low, high = np.quantile(values[sample_indices].mean(axis=1), [.025, .975], axis=0)
        layers = np.arange(len(avg))
        ax.fill_between(layers, low, high, color=color, alpha=.16, linewidth=0)
        ax.plot(layers, avg, color=color, lw=1.35, ls=pattern, label=label)
    ax.set_xlabel('Center layer (zero-based)')
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 10, 20, 30, 40, 47])
axs[0].set_ylabel('Recovered probability\n(percentage points)')
axs[1].legend(frameon=False, loc='upper left')
fig.savefig(output / 'tracing_profiles.pdf')
plt.close(fig)


groups = ["first_subject", "middle_subject", "last_subject",
          "first_after", "middle_after", "last_prompt"]
labels = ["First subject token", "Middle subject tokens", "Last subject token",
          "First after subject", "Other later tokens", "Last prompt token"]
effects = [100 * np.array([summary["traces"][kind][group]["effect"] for group in groups])
           for kind in ("mlp", "attn")]
fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.8), layout="constrained", sharex=True)
scale = max(np.abs(values).max() for values in effects)
for ax, values, title in zip(axes, effects, ["MLP restoration", "Attention restoration"]):
    heatmap = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-scale,
                        vmax=scale, interpolation="nearest")
    ax.set_yticks(range(6), labels, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
axes[-1].set_xlabel("Center layer of the ten-layer restoration window")
fig.colorbar(heatmap, ax=axes, label="Recovery in answer probability (percentage points)", shrink=.88)
fig.savefig(output / "tracing.pdf")
plt.close(fig)
