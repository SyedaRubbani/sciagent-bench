# Running the GPU experiments (Kaggle or Colab)

All GPU-dependent results come from one resumable command, `sciagent experiments`, wrapped in a notebook.

## Kaggle (recommended: free 2 x T4)

1. Push this repository to GitHub, or upload it as a Kaggle *dataset* (zip of the `sciagent-bench` folder).
2. Create a notebook from `notebooks/kaggle_gpu_experiments.ipynb` (File -> Import notebook).
3. In the right-hand panel set **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: Files only**.
4. Choose `PROFILE = "kaggle-t4"` (about 6 hours) or `"quick"` (about 1.5 hours) and *Run all*, or
   *Save & Run All (Commit)* to run it in the background within Kaggle's 12-hour limit.
5. Download `results_bundle.zip` from the output panel.

If the session stops, re-run the notebook: steps with a `DONE.json` are skipped.

## Colab

Open `notebooks/colab_gpu_experiments.ipynb`, pick a GPU runtime, run all. Results go to Google Drive. An L4 or
A100 runs the `kaggle-t4` profile faster; on a T4 use `quick`.

## What runs where

| step | GPU use |
|---|---|
| tests | torch kernel tests on CUDA |
| verify | 20M-draw Monte-Carlo and PCG heat solves up to 2048 x 2048 on the GPU |
| gpu_bench | kernel throughput, NumPy/SciPy vs torch-CUDA |
| bench_cpu | none (CPU planners; included so one bundle holds everything) |
| surrogate | surrogate training on the GPU; latency is still measured on CPU, the deployment target |
| bench_llm | teacher LLM on GPU 0 (vLLM, or transformers if vLLM cannot be installed) |
| traces | teacher LLM on GPU 0, training seeds 1000+ (disjoint from evaluation seeds 0-9) |
| train_student | LoRA fine-tuning of the 0.5B student on GPU 1 (fp16 AMP on T4, bf16 on Ampere+) |
| inference | each model served alone; latency, throughput, valid-JSON rate |
| bench_distill | base and distilled students on GPU 1, held-out seeds |

## Serving stack

`sciagent.serving.launch.start_server` first tries vLLM, which `ensure_vllm` installs into its own virtualenv so the
notebook's PyTorch is never replaced. If vLLM is unavailable or fails to start, the step falls back to
`sciagent.serving.hf_server`, an OpenAI-compatible transformers server with dynamic request batching (and
bitsandbytes 4-bit loading). The backend used is recorded with every result. AWQ teachers need vLLM; with the
fallback the profile switches to the non-quantised `hf_fallback_model`.

## Merging GPU and CPU results

```bash
unzip results_bundle.zip -d results/gpu
sciagent report results/cpu results/gpu --out docs/TECHNICAL_REPORT.md
```

## Checking the pipeline without a GPU

```bash
sciagent experiments --profile cpu-smoke --results /tmp/smoke
```

This runs every step with a fake OpenAI-compatible teacher (the scripted policy over HTTP) and a tiny randomly
initialised student, so the plumbing (servers, traces, SFT data, LoRA training, merging, serving, benchmarking,
report) is exercised end to end in a few minutes. CI runs it on every push.
