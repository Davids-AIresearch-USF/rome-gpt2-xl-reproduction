# Reproducing ROME on GPT-2 XL

This CAI 5307 project compares Rank-One Model Editing (ROME), unconstrained fine-tuning, constrained fine-tuning, and the original GPT-2 XL on the same 200 CounterFact records. It also measures the effect of adding neutral text to a prompt and performs causal tracing on 50 separate factual statements.

## Recalculate the results and draw the figures

Run these commands from the repository directory. They use the saved predictions and traces and work on a CPU. Choose new output names if `summary.json` or `plots` already exists.

```sh
python -m pip install numpy matplotlib
python analyze_rome.py results/ROME_Experiment_Results.zip summary.json
python plot_results.py results/ROME_Experiment_Results.zip summary.json plots
```

The summary uses 200 evaluation cases and 50 traced statements. The 20 pilot cases are separate.

The report source and the three figures are in `paper`. To build the report with an installed LaTeX distribution, run `pdflatex main.tex`, `bibtex main`, and `pdflatex main.tex` twice more from that directory.

## Run the experiment

1. Open `ROME_GPT_2_XL_Reproduction.ipynb` in a fresh Google Colab GPU runtime and run the four cells in order. The 200 evaluation cases took 2.38 hours on a Tesla T4.
2. The first cell installs the Python packages and downloads the datasets, [ROME code](https://github.com/kmeng01/rome/tree/0874014cd9837e4365f3e6f3c71400ef11509e04), and [GPT-2 XL checkpoint](https://huggingface.co/openai-community/gpt2-xl/tree/15ea56dee5df4983c59b2538573817e1667135e2) used in the experiment.
3. Results go in `/content/rome_outputs`. To resume after an interruption, restore that whole folder from a single checkpoint in a fresh GPU runtime before running the four cells. The notebook checks the saved plans, reuses the saved ROME editing contexts, and skips completed cases and traces.
4. Run the following in a new Colab code cell to download a checkpoint or the final results. Use the downloaded ZIP as the input to the CPU commands above.

```python
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from google.colab import files

folder = Path("/content/rome_outputs")
archive = folder.parent / f"ROME_Results_{datetime.now():%Y%m%d_%H%M%S}.zip"
with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
    for path in sorted(folder.iterdir()):
        if path.suffix in {".json", ".npz"}:
            bundle.write(path, path.relative_to(folder.parent))
files.download(str(archive))
```

## Sources

- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), NeurIPS 2022.
- [Authors' ROME implementation](https://github.com/kmeng01/rome) and [dataset links](https://rome.baulab.info/).
