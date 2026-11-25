# Reproducing ROME on GPT-2 XL

This CAI 5307 project compares Rank-One Model Editing (ROME), unconstrained fine-tuning, constrained fine-tuning, and the original GPT-2 XL on the same 200 CounterFact records. It also measures the effect of adding neutral text to a prompt and performs causal tracing on 50 separate factual statements.

## Recalculate the results and draw the figures

From this directory, run the commands below with Python. Choose new output names if `summary.json` or the `plots` directory already exists.

```sh
python -m pip install numpy matplotlib
python analyze_rome.py results/ROME_Experiment_Results.zip summary.json
python plot_results.py results/ROME_Experiment_Results.zip summary.json plots
```

The ZIP contains 20 pilot records, 200 evaluation records, 50 MLP and 50 attention arrays, the case and tracing plans, and the ROME rewrite contexts. The analysis checks case identities, keeps pilot records out of the evaluation, and writes the editing scores, paired bootstrap intervals and mean tracing effects. The plotting command makes three PDFs showing the editing scores, the two tracing profiles, and the tracing heat maps.

The report source and the three figures are in `paper`. To build the report with an installed LaTeX distribution, run `pdflatex main.tex`, `bibtex main`, and `pdflatex main.tex` twice more from that directory.

## Run the experiment

1. Open `ROME_GPT_2_XL_Reproduction.ipynb` in Google Colab and select a GPU runtime. The reported experiment used an NVIDIA Tesla T4. The 200 editing records took 2.38 GPU hours, with setup, pilot and tracing time on top of that.
2. Run all four cells, in order. The first cell checks out the authors' code at `0874014cd9837e4365f3e6f3c71400ef11509e04`, downloads CounterFact and the known-facts file, and loads `openai-community/gpt2-xl` at revision `15ea56dee5df4983c59b2538573817e1667135e2`. Use a fresh runtime for a full rerun. The notebook installs its Python dependencies. It uses the authors' published cached second moment for ROME.
3. The second cell chooses the pilot and evaluation records and records the context prefixes. The third cell runs the pilot and the four model conditions, restores the weights after each edit, and writes one JSON file per case to `/content/rome_outputs`. The fourth cell writes the 100 tracing arrays. Check that the folder contains `plan.json`, `trace_plan.json`, `edit_contexts.json`, and all expected case and trace files.
4. In Colab's Terminal, make a ZIP from `/content` with the `rome_outputs` folder at its root. Download the ZIP from the Files tab, then analyze it using the command above with the downloaded ZIP path.

```sh
cd /content
python -m zipfile -c ROME_Experiment_Results.zip rome_outputs
```

Colab stores `/content` on the current runtime. When continuing the same run after a runtime replacement, restore a saved `rome_outputs` folder before running the cells. Existing case and trace files are skipped. Keep outputs from different versions in separate folders. GPU and library differences can produce small numerical changes.

## Sources

- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), NeurIPS 2022.
- [Authors' ROME implementation](https://github.com/kmeng01/rome) and [dataset links](https://rome.baulab.info/).
