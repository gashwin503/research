from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Model
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float16"          # stored as string for JSON serialisation
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # MathNeuro
    target_layer_suffix: str = "mlp.down_proj"
    top_k_ratio: float = 0.01
    n_calibration_samples: int = 200
    n_eval_samples: int = 100
    n_random_trials: int = 3
    seed: int = 0

    # Runtime / tracking
    batch_size: int = 8
    max_length: int = 512
    max_new_tokens: int = 200
    project: str = "mathneuro-finance-pilot"
    run_name: Optional[str] = None
    offline: bool = False
    output_dir: str = "runs"
    save_masks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep torch_dtype human-readable
        return d

    def resolve_dtype(self) -> torch.dtype:
        return getattr(torch, self.torch_dtype)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic (may slow things down; comment out if needed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hash_list(items: List[str], n_chars: int = 16) -> str:
    """Stable content hash of a list of strings."""
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:n_chars]


def get_model_provenance(model_id: str) -> Dict[str, Any]:
    """Best-effort model revision / commit info from the Hub."""
    info: Dict[str, Any] = {"model_id": model_id}
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        info["sha"] = model_info.sha
        info["last_modified"] = str(getattr(model_info, "lastModified", None))
        info["tags"] = list(getattr(model_info, "tags", []) or [])
        info["pipeline_tag"] = getattr(model_info, "pipeline_tag", None)
    except Exception as e:
        info["error"] = str(e)
    return info


def get_dataset_fingerprint(ds) -> str:
    """HuggingFace Datasets fingerprint (content-addressed)."""
    try:
        return ds._fingerprint
    except Exception:
        return "unknown"


def environment_info() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hostname": platform.node(),
    }


# ---------------------------------------------------------------------------
# Core MathNeuro classes (adapted from original)
# ---------------------------------------------------------------------------

class MathNeuroAnalyzer:
    def __init__(self, model, config: PipelineConfig, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name.endswith(self.config.target_layer_suffix):
                hook = module.register_forward_pre_hook(self._get_hook_fn(name))
                self.hooks.append(hook)

    def _get_hook_fn(self, name):
        def hook(module, args):
            x = args[0].detach().float()
            mask = self._current_mask.to(x.device).unsqueeze(-1)
            x = x * mask
            sq_sum = (x ** 2).sum(dim=(0, 1))
            if name not in self._acc:
                self._acc[name] = sq_sum
            else:
                self._acc[name] += sq_sum
        return hook

    def compute_importance_scores(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        self._acc = {}
        importance_scores = {}

        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i : i + self.config.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.config.device)
            self._current_mask = inputs["attention_mask"].float()
            with torch.no_grad():
                self.model(**inputs)

        for name, module in self.model.named_modules():
            if name in self._acc:
                w_abs = module.weight.data.abs().float()
                act_norm = self._acc[name].sqrt()
                importance_scores[name] = w_abs * act_norm.unsqueeze(0)

        return importance_scores

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def isolate_task_parameters(
    analyzer: MathNeuroAnalyzer,
    task_prompts: List[str],
    control_prompts: List[str],
    top_k_ratio: float,
) -> Dict[str, torch.Tensor]:
    task_scores = analyzer.compute_importance_scores(task_prompts)
    control_scores = analyzer.compute_importance_scores(control_prompts)
    isolated_masks = {}

    for layer in task_scores:
        t_score, c_score = task_scores[layer], control_scores[layer]

        def topk_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
            k = max(int(score.numel() * ratio), 1)
            flat = score.reshape(-1)
            idx = torch.topk(flat, k, sorted=False).indices
            m = torch.zeros_like(flat, dtype=torch.bool)
            m[idx] = True
            return m.view_as(score)

        top_task_mask = topk_mask(t_score, top_k_ratio)
        top_control_mask = topk_mask(c_score, top_k_ratio)
        isolated_masks[layer] = top_task_mask & (~top_control_mask)

    return isolated_masks


def apply_mask(model, masks: Dict[str, torch.Tensor], factor: float = 0.0):
    saved = {}
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name].to(module.weight.device)
            saved[name] = module.weight.data.clone()
            module.weight.data[mask] = module.weight.data[mask] * factor
    return saved


def restore_weights(model, saved):
    for name, module in model.named_modules():
        if name in saved:
            module.weight.data.copy_(saved[name])


def random_masks_like(masks: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    random_masks = {}
    for name, mask in masks.items():
        n = int(mask.sum().item())
        flat = torch.zeros(mask.numel(), dtype=torch.bool)
        idx = torch.randperm(mask.numel(), generator=g)[:n]
        flat[idx] = True
        random_masks[name] = flat.view(mask.shape)
    return random_masks


def extract_answer(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def eval_gsm8k(
    model,
    tokenizer,
    dataset,
    device: str,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> float:
    tokenizer.padding_side = "left"
    correct = 0
    total = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        questions = batch["question"]
        golds = [a.split("####")[-1].strip().replace(",", "") for a in batch["answer"]]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for pred_text, gold in zip(gen, golds):
            pred = extract_answer(pred_text)
            total += 1
            if pred is not None:
                try:
                    if abs(float(pred) - float(gold)) < 1e-4:
                        correct += 1
                except ValueError:
                    pass
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def build_prompts(config: PipelineConfig):
    """Load datasets and build the four prompt sets used by MathNeuro."""
    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k_train = (
        gsm8k["train"].shuffle(seed=config.seed).select(range(config.n_calibration_samples))
    )
    gsm8k_test = (
        gsm8k["test"].shuffle(seed=config.seed).select(range(config.n_eval_samples))
    )

    wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    wikitext_lines = [t for t in wikitext["train"]["text"] if len(t.split()) > 8]
    random.shuffle(wikitext_lines)
    language_prompts = wikitext_lines[: config.n_calibration_samples]

    math_prompts = [
        f"Question: {row['question']}\nAnswer: {row['answer']}" for row in gsm8k_train
    ]

    meta = {
        "gsm8k": {
            "name": "openai/gsm8k",
            "config": "main",
            "train_fingerprint": get_dataset_fingerprint(gsm8k["train"]),
            "test_fingerprint": get_dataset_fingerprint(gsm8k["test"]),
            "n_calibration": len(gsm8k_train),
            "n_eval": len(gsm8k_test),
        },
        "wikitext": {
            "name": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "train_fingerprint": get_dataset_fingerprint(wikitext["train"]),
            "n_language_prompts": len(language_prompts),
        },
        "prompt_hashes": {
            "math_prompts": hash_list(math_prompts),
            "language_prompts": hash_list(language_prompts),
        },
        "token_stats": {},  # filled after tokenizer is available
    }
    return math_prompts, language_prompts, gsm8k_test, meta


def run_experiment(config: PipelineConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    run_dir = Path(config.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- Config snapshot -----
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # ----- Environment & model provenance -----
    env = environment_info()
    model_prov = get_model_provenance(config.model_id)

    # ----- HF login (optional) -----
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token.strip())

    # ----- Load model & tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.resolve_dtype(),
        device_map="auto" if config.device == "cuda" else None,
    )
    if config.device == "cpu":
        model = model.to("cpu")
    model.eval()

    # ----- Datasets & prompts -----
    math_prompts, language_prompts, gsm8k_test, data_meta = build_prompts(config)

    # Token-length stats for reproducibility checks
    math_tok_lens = [len(tokenizer.encode(p)) for p in math_prompts]
    lang_tok_lens = [len(tokenizer.encode(p)) for p in language_prompts]
    data_meta["token_stats"] = {
        "math_mean_tokens": sum(math_tok_lens) / len(math_tok_lens),
        "math_max_tokens": max(math_tok_lens),
        "language_mean_tokens": sum(lang_tok_lens) / len(lang_tok_lens),
        "language_max_tokens": max(lang_tok_lens),
    }

    # ----- wandb -----
    wandb_run = None
    if WANDB_AVAILABLE and not config.offline:
        wandb_run = wandb.init(
            project=config.project,
            name=config.run_name,
            config=config.to_dict(),
            dir=str(run_dir),
            reinit=True,
        )
        wandb.config.update(
            {
                "environment": env,
                "model_provenance": model_prov,
                "dataset_meta": data_meta,
            }
        )
    elif WANDB_AVAILABLE and config.offline:
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            project=config.project,
            name=config.run_name,
            config=config.to_dict(),
            dir=str(run_dir),
            reinit=True,
        )
        wandb.config.update(
            {
                "environment": env,
                "model_provenance": model_prov,
                "dataset_meta": data_meta,
            }
        )

    # Save local metadata
    meta = {
        "config": config.to_dict(),
        "environment": env,
        "model_provenance": model_prov,
        "dataset_meta": data_meta,
        "run_dir": str(run_dir),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    results: Dict[str, Any] = {}

    # ----- Baseline accuracy -----
    print("Evaluating baseline accuracy...")
    baseline_acc = eval_gsm8k(
        model,
        tokenizer,
        gsm8k_test,
        device=config.device,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    results["baseline_accuracy"] = baseline_acc
    print(f"  Baseline GSM8K accuracy: {baseline_acc:.4f}")

    if wandb_run:
        wandb.log({"baseline_accuracy": baseline_acc})

    # ----- Isolate parameters -----
    print("Computing importance scores & isolating math parameters...")
    analyzer = MathNeuroAnalyzer(model, config, tokenizer)
    isolated_masks = isolate_task_parameters(
        analyzer, math_prompts, language_prompts, config.top_k_ratio
    )
    analyzer.remove_hooks()

    n_params_isolated = sum(int(m.sum().item()) for m in isolated_masks.values())
    results["n_params_isolated"] = n_params_isolated
    print(f"  Isolated {n_params_isolated} parameters across {len(isolated_masks)} layers")

    if config.save_masks:
        mask_path = run_dir / "isolated_masks.pt"
        torch.save({k: v.cpu() for k, v in isolated_masks.items()}, mask_path)
        if wandb_run:
            wandb.save(str(mask_path))

    # ----- Targeted pruning -----
    print("Applying targeted (math-isolated) pruning...")
    saved = apply_mask(model, isolated_masks, factor=0.0)
    targeted_acc = eval_gsm8k(
        model,
        tokenizer,
        gsm8k_test,
        device=config.device,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
    )
    restore_weights(model, saved)
    results["targeted_pruning_accuracy"] = targeted_acc
    results["targeted_drop"] = baseline_acc - targeted_acc
    print(f"  Targeted pruning accuracy: {targeted_acc:.4f} (drop {results['targeted_drop']:.4f})")

    if wandb_run:
        wandb.log(
            {
                "targeted_pruning_accuracy": targeted_acc,
                "targeted_drop": results["targeted_drop"],
            }
        )

    # ----- Random controls -----
    random_accs = []
    for trial in range(config.n_random_trials):
        print(f"Random pruning trial {trial + 1}/{config.n_random_trials}...")
        rmasks = random_masks_like(isolated_masks, seed=config.seed + 1000 + trial)
        saved = apply_mask(model, rmasks, factor=0.0)
        racc = eval_gsm8k(
            model,
            tokenizer,
            gsm8k_test,
            device=config.device,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
        )
        restore_weights(model, saved)
        random_accs.append(racc)
        if wandb_run:
            wandb.log({f"random_pruning_accuracy_trial_{trial}": racc})

    results["random_pruning_accuracies"] = random_accs
    results["random_pruning_mean"] = sum(random_accs) / len(random_accs)
    results["random_drop_mean"] = baseline_acc - results["random_pruning_mean"]
    print(
        f"  Random pruning mean accuracy: {results['random_pruning_mean']:.4f} "
        f"(drop {results['random_drop_mean']:.4f})"
    )

    if wandb_run:
        wandb.log(
            {
                "random_pruning_mean": results["random_pruning_mean"],
                "random_drop_mean": results["random_drop_mean"],
                "n_params_isolated": n_params_isolated,
            }
        )

    # ----- Final summary -----
    summary = {
        **results,
        "config": config.to_dict(),
        "model_provenance": model_prov,
        "dataset_meta": data_meta,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if wandb_run:
        wandb.summary.update(results)
        wandb.finish()

    print(f"\nRun artifacts saved to: {run_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="MathNeuro experiment with tracking")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_calibration_samples", type=int, default=200)
    parser.add_argument("--n_eval_samples", type=int, default=100)
    parser.add_argument("--top_k_ratio", type=float, default=0.01)
    parser.add_argument("--n_random_trials", type=int, default=3)
    parser.add_argument("--project", type=str, default="mathneuro-finance-pilot")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--offline", action="store_true", help="Force wandb offline mode")
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--no_save_masks", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_id=args.model_id,
        seed=args.seed,
        n_calibration_samples=args.n_calibration_samples,
        n_eval_samples=args.n_eval_samples,
        top_k_ratio=args.top_k_ratio,
        n_random_trials=args.n_random_trials,
        project=args.project,
        run_name=args.run_name,
        offline=args.offline,
        output_dir=args.output_dir,
        save_masks=not args.no_save_masks,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    print("=== MathNeuro Experiment (tracked) ===")
    print(json.dumps(config.to_dict(), indent=2))
    run_experiment(config)
