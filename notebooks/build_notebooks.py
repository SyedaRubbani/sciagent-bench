"""Generate the Kaggle and Colab GPU notebooks from one definition (run: python notebooks/build_notebooks.py)."""

import json
from pathlib import Path

REPO = "https://github.com/SyedaRubbani/sciagent-bench.git"


def nb(platform: str) -> dict:
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

    if platform == "kaggle":
        md("""
# SciAgent-Bench: GPU experiments (Kaggle)

Runs every GPU-dependent experiment of the benchmark and writes a self-contained results bundle.

**Notebook settings (right-hand panel):** Accelerator **GPU T4 x2**, Internet **On**, Persistence **Files only**
(so an interrupted session can resume; finished steps are skipped).

| step | what it measures | T4 x2 time (full / quick) |
|---|---|---|
| tests | the full test suite, with the torch kernels on the GPU | 3 min |
| verify | reference optima: 20M-draw Monte-Carlo, heat grids to 2048², optimiser robustness | 15 / 5 min |
| gpu_bench | NumPy/SciPy vs torch-CUDA kernel throughput | 5 / 2 min |
| bench_cpu | classical planners, 20 seeds x 2 budgets, statistics | 20 / 5 min |
| surrogate | data curation, GPU training, distillation, int8 | 15 / 3 min |
| bench_llm | open-model LLM planner (with / without GP advisor) vs baselines | 90 / 20 min |
| traces | teacher planning traces for distillation | 90 / 15 min |
| train_student | LoRA distillation of a 0.5B planner on the second T4 | 30 / 10 min |
| inference | latency, throughput, JSON validity: teacher vs base vs distilled (fp16, 4-bit) | 15 / 5 min |
| bench_distill | planning quality of the distilled model on held-out seeds | 60 / 15 min |
| report | technical report + zip bundle | 1 min |

Set `PROFILE = "quick"` for a ~1.5 h run that exercises everything with fewer seeds and a 1.5B teacher.
""")
        code(f"""
PROFILE = "kaggle-t4"          # or "quick"
REPO_URL = "{REPO}"
RESULTS = "/kaggle/working/results"
import os, glob, subprocess, sys
src = sorted(glob.glob("/kaggle/input/*/sciagent-bench")) or sorted(glob.glob("/kaggle/input/sciagent-bench*"))
if src:   # code uploaded as a Kaggle dataset
    !cp -r {{src[0]}} /kaggle/working/sciagent-bench
elif not os.path.exists("/kaggle/working/sciagent-bench"):
    !git clone -q {{REPO_URL}} /kaggle/working/sciagent-bench
%cd /kaggle/working/sciagent-bench
os.environ["SCIAGENT_VLLM_VENV"] = "/tmp/vllm-env"   # vLLM gets its own venv (not saved to output); the kernel torch is untouched
""")
    else:
        md("""
# SciAgent-Bench: GPU experiments (Colab)

Runtime -> Change runtime type -> **GPU** (L4 or A100 recommended; T4 works with `PROFILE = "quick"`). Results are
written to Google Drive so an interrupted session can resume; finished steps are skipped.
""")
        code(f"""
PROFILE = "quick"              # "kaggle-t4" settings also work on an L4/A100
REPO_URL = "{REPO}"
from google.colab import drive
drive.mount("/content/drive")
RESULTS = "/content/drive/MyDrive/sciagent_results"
import os, subprocess, sys
if not os.path.exists("/content/sciagent-bench"):
    !git clone -q {{REPO_URL}} /content/sciagent-bench
%cd /content/sciagent-bench
os.environ["SCIAGENT_VLLM_VENV"] = "/content/vllm-env"
""")
    code("""
!nvidia-smi -L
!pip install -q -e ".[dev,surrogate,mcp,plot]" "transformers>=4.44" "peft>=0.12" accelerate bitsandbytes hypothesis
""")
    md("## Run the pipeline\nEach cell is one resumable step. Outputs go to `RESULTS/<step>/`; failures are written to `ERROR.txt` "
       "and do not stop later independent steps.")
    for step, note in [("env tests", "Environment record and the test suite (torch kernels on CUDA)."),
                       ("verify", "Reference solutions with uncertainty (GPU Monte-Carlo and matrix-free PCG)."),
                       ("gpu_bench", "Kernel throughput."),
                       ("bench_cpu", "Classical planners: 20 seeds, two budgets, bootstrap CIs and paired tests."),
                       ("surrogate", "Surrogate: curated data, GPU training, distillation, int8."),
                       ("bench_llm", "Open-model LLM planner benchmark (vLLM if it installs, transformers otherwise)."),
                       ("traces", "Teacher traces on training seeds 1000+ (disjoint from evaluation seeds)."),
                       ("train_student", "LoRA distillation of the 0.5B student."),
                       ("inference", "Serving efficiency: teacher, base, distilled fp16 and 4-bit."),
                       ("bench_distill", "Planning quality of the distilled model on held-out seeds."),
                       ("report", "Technical report and results bundle.")]:
        md(f"### {step}\n{note}")
        code(f"!sciagent experiments --profile $PROFILE --results $RESULTS --steps {step}")
    md("## Read the report")
    code("""
from IPython.display import Markdown, display
display(Markdown(open(f"{RESULTS}/report/TECHNICAL_REPORT.md").read()))
print(open(f"{RESULTS}/experiments_status.json").read())
""")
    md("The bundle `results_bundle.zip` (next to the results folder) contains every raw result, trace, cassette and log. "
       "Download it and run `sciagent report results/cpu <unzipped>/results --out docs/TECHNICAL_REPORT.md` in the repo "
       "to merge GPU and CPU results into one report.")
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if platform == "kaggle":
        meta["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}
    else:
        meta["accelerator"] = "GPU"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


here = Path(__file__).parent
(here / "kaggle_gpu_experiments.ipynb").write_text(json.dumps(nb("kaggle"), indent=1))
(here / "colab_gpu_experiments.ipynb").write_text(json.dumps(nb("colab"), indent=1))
print("wrote", here / "kaggle_gpu_experiments.ipynb", here / "colab_gpu_experiments.ipynb")
