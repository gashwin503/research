import torch
import re
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from google.colab import userdata


class PipelineConfig:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.torch_dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_layer_suffix = "mlp.down_proj"
        self.top_k_ratio = 0.01
        self.n_calibration_samples = 200
        self.n_eval_samples = 100
        self.n_random_trials = 3
        self.seed = 0


config = PipelineConfig()
random.seed(config.seed)
torch.manual_seed(config.seed)

hf_token = userdata.get('HF_TOKEN')
if hf_token:
    login(token=hf_token.strip())

tokenizer = AutoTokenizer.from_pretrained(config.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    torch_dtype=config.torch_dtype,
    device_map="auto"
)
model.eval()

gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_train = gsm8k["train"].shuffle(seed=config.seed).select(range(config.n_calibration_samples))
gsm8k_test = gsm8k["test"].shuffle(seed=config.seed).select(range(config.n_eval_samples))

wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
wikitext_lines = [t for t in wikitext["train"]["text"] if len(t.split()) > 8]
random.shuffle(wikitext_lines)
language_prompts = wikitext_lines[:config.n_calibration_samples]

math_prompts = [f"Question: {row['question']}\nAnswer: {row['answer']}" for row in gsm8k_train]


class MathNeuroAnalyzer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
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

    def compute_importance_scores(self, prompts, batch_size=8, max_length=512):
        self._acc = {}
        importance_scores = {}

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(self.config.device)
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


def isolate_task_parameters(analyzer, task_prompts, control_prompts, top_k_ratio=0.01):
    task_scores = analyzer.compute_importance_scores(task_prompts)
    control_scores = analyzer.compute_importance_scores(control_prompts)
    isolated_masks = {}

    for layer in task_scores:
        t_score, c_score = task_scores[layer], control_scores[layer]

        def topk_mask(score, ratio):
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


def apply_mask(model, masks, factor=0.0):
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


def random_masks_like(masks, seed):
    g = torch.Generator().manual_seed(seed)
    random_masks = {}
    for name, mask in masks.items():
        n = int(mask.sum().item())
        flat = torch.zeros(mask.numel(), dtype=torch.bool)
        idx = torch.randperm(mask.numel(), generator=g)[:n]
        flat[idx] = True
        random_masks[name] = flat.view(mask.shape)
    return random_masks


def extract_answer(text):
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def eval_gsm8k(model, tokenizer, dataset, batch_size=8, max_new_tokens=200):
    tokenizer.padding_side = "left"
    correct = 0
    total = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        questions = batch["question"]
        golds = [a.split("####")[-1].strip().replace(",", "") for a in batch["answer"]]
        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(config.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for pred_text, gold in zip(gen, golds):
            pred = extract_answer(pred_text)
            total += 1
            if pred is not None:
                try:
                    if abs(float(pred) - float(gold)) < 1e-4:
                        correct += 1
                except ValueError:
                    pass
    return correct / total
