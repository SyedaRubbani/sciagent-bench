"""LoRA-distil a teacher planner into a small open model (GPU; tested target: one 16-24 GB GPU).

    pip install "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" datasets
    python train_lora.py --data sft.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --out planner-lora

Loss is computed on the assistant (plan) tokens only. After training the adapter is merged into the base
weights (``<out>/merged``) so the model can be served by vLLM like any other checkpoint.
"""

import argparse
import inspect
import json
import math

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="sft.jsonl")
ap.add_argument("--val", default=None)
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", default="planner-lora")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max-len", type=int, default=6144)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=-1, help="cap optimiser steps (smoke tests)")
a = ap.parse_args()
val_path = a.val or a.data.replace(".jsonl", ".val.jsonl")

tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()   # A100/L4/H100; T4 and P100 are fp16-only
fp16 = cuda and not bf16                          # fp16 AMP needs fp32 master weights
dtype = torch.bfloat16 if bf16 else torch.float32
kw = {"device_map": "auto"} if cuda else {}
try:
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype, **kw)
except TypeError:  # transformers < 4.56
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, **kw)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def encode(ex):
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + ex["messages"][-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > a.max_len:  # never truncate: a cut-off plan teaches the model to emit broken JSON
        return None
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels}


def load(path):
    rows = [encode(json.loads(line)) for line in open(path) if line.strip()]
    kept = [r for r in rows if r is not None]
    print(f"{path}: kept {len(kept)} / {len(rows)} examples (dropped {len(rows) - len(kept)} longer than {a.max_len} tokens)")
    return kept


train_ds, val_ds = load(a.data), load(val_path)
print(f"train {len(train_ds)}  val {len(val_ds)}")


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


steps_per_epoch = math.ceil(len(train_ds) / (a.batch * a.grad_accum))
total_steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * a.epochs)
targs = dict(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.batch,
             gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine",
             warmup_steps=max(1, int(0.05 * total_steps)), logging_steps=5, save_strategy="no", bf16=bf16, fp16=fp16,
             report_to="none", remove_unused_columns=False, seed=a.seed, data_seed=a.seed, max_steps=a.max_steps)
accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
targs["eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"] = "epoch"  # renamed in 4.41
args = TrainingArguments(**{k: v for k, v in targs.items() if k in accepted})
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
trainer.train()
model.save_pretrained(f"{a.out}/adapter")
merged = model.merge_and_unload()
if cuda:
    merged = merged.to(torch.bfloat16 if bf16 else torch.float16)  # serve in half precision
merged.save_pretrained(f"{a.out}/merged", safe_serialization=True)
tok.save_pretrained(f"{a.out}/merged")
hist = trainer.state.log_history
json.dump({"log_history": hist, "train": len(train_ds), "val": len(val_ds), "base": a.base, "rank": a.rank,
           "epochs": a.epochs, "lr": a.lr, "bf16": bf16, "fp16": fp16},
          open(f"{a.out}/train_summary.json", "w"), indent=2)
print(json.dumps(hist[-3:], indent=2))
