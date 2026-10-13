"""LoRA-distil a teacher planner into a small open model (GPU; tested target: one 16-24 GB GPU).

    pip install "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" datasets
    python train_lora.py --data sft.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --out planner-lora

Loss is computed on the assistant (plan) tokens only. After training the adapter is merged into the base
weights (``<out>/merged``) so the model can be served by vLLM like any other checkpoint.
"""

import argparse
import json

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
a = ap.parse_args()
val_path = a.val or a.data.replace(".jsonl", ".val.jsonl")

tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, device_map="auto")
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
    ids = tok(full, add_special_tokens=False)["input_ids"][: a.max_len]
    labels = ([-100] * len(p_ids) + ids[len(p_ids):])[: len(ids)]
    return {"input_ids": ids, "labels": labels}


def load(path):
    return [encode(json.loads(line)) for line in open(path) if line.strip()]


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


args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                         per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                         lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=5, eval_strategy="epoch",
                         save_strategy="epoch", bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                         report_to="none", remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
trainer.train()
model.save_pretrained(f"{a.out}/adapter")
merged = model.merge_and_unload()
merged.save_pretrained(f"{a.out}/merged", safe_serialization=True)
tok.save_pretrained(f"{a.out}/merged")
print(json.dumps(trainer.state.log_history[-3:], indent=2))
