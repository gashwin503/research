# Experiment Tracking & Reproducibility

This directory contains a fully tracked version of the MathNeuro-style pipeline (`eval_tracked.py`) ready for the TEAM KHAT research on separating mathematical reasoning from financial knowledge.

## What is logged for every run

| Category | What is recorded | Where |
|----------|------------------|-------|
| **Config** | Full hyper-parameters (model, seed, top-k, sample counts, batch size, etc.) | `runs/<timestamp>/config.json` + wandb config |
| **Model** | `model_id`, Hub commit SHA, last-modified, tags | `run_meta.json` + wandb |
| **Datasets** | Dataset name + config, HuggingFace fingerprints, prompt content hashes, token-length statistics | `run_meta.json` + wandb |
| **Environment** | Python/torch versions, CUDA availability, hostname | `run_meta.json` + wandb |
| **Metrics** | Baseline accuracy, targeted-pruning accuracy & drop, random-control accuracies | `results.json` + wandb |
| **Artifacts** | Isolated parameter masks (`.pt`) | `runs/<timestamp>/isolated_masks.pt` (also uploaded to wandb when online) |

## Quick start

```bash
# Optional: Hugging Face token (for gated models)
export HF_TOKEN=hf_...

# Optional: Weights & Biases (omit or use --offline for local-only runs)
export WANDB_API_KEY=...

# Dry-run / offline (no cloud upload)
python eval_tracked.py --offline --n_eval_samples 20 --n_calibration_samples 40

# Full pilot
python eval_tracked.py --project mathneuro-finance-pilot --run_name pilot-qwen05b
```

## Key CLI flags

- `--model_id` – any open-weight causal LM on the Hub
- `--seed` – controls all RNGs (Python, torch, dataset shuffle)
- `--n_calibration_samples` / `--n_eval_samples`
- `--top_k_ratio` – fraction of parameters kept as “important”
- `--offline` – force wandb offline mode (still writes local files)
- `--output_dir` – root for timestamped run folders (default: `runs/`)

## Design notes for the finance extension

The current script reproduces the classic MathNeuro math-vs-language isolation (GSM8K vs WikiText).  
When you add the matched finance / pure-math / non-finance / nonsense conditions described in the proposal:

1. Extend `build_prompts()` to emit the four condition lists.
2. Add each condition’s fingerprint / content hash to `dataset_meta`.
3. Log per-condition baseline accuracies and per-condition pruning drops.
4. Keep the same `PipelineConfig` + wandb schema so every new condition is automatically versioned.

All random operations are seeded from `config.seed`, and every derived prompt set is content-hashed, so re-running with the same config on the same Hub model revision yields identical masks (up to floating-point non-determinism on GPU).

## Local run layout

```
runs/
  20260928_194512/
    config.json          # exact CLI + defaults used
    run_meta.json        # env + model SHA + dataset fingerprints
    results.json         # all numeric metrics
    isolated_masks.pt    # torch tensors of the isolated masks
```
