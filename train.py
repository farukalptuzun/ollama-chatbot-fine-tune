print("🚀 train.py başlatılıyor...", flush=True)

import yaml
import argparse
import os
import math
import sys

print("📚 Kütüphaneler yükleniyor...", flush=True)
import torch
print("✅ torch yüklendi", flush=True)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
print("✅ transformers yüklendi", flush=True)

from datasets import load_dataset
print("✅ datasets yüklendi", flush=True)

# ---------- args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", required=True)
parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--dataset", default="data/train.jsonl")
args = parser.parse_args()

# ---------- load run config ----------
if not os.path.exists("runs.yaml"):
    print("❌ runs.yaml bulunamadı!")
    sys.exit(1)

with open("runs.yaml") as f:
    RUNS = yaml.safe_load(f)

if args.run_id not in RUNS:
    print(f"❌ RUN ID '{args.run_id}' runs.yaml'de bulunamadı!")
    print(f"Mevcut ID'ler: {list(RUNS.keys())}")
    sys.exit(1)

cfg = RUNS[args.run_id]

# ---------- dataset kontrolü ----------
if not os.path.exists(args.dataset):
    print(f"❌ Dataset dosyası bulunamadı: {args.dataset}")
    sys.exit(1)

# ---------- tokenizer ----------
print(f"📥 Tokenizer yükleniyor: {args.model_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )
except Exception as e:
    print(f"❌ Tokenizer yüklenemedi: {e}")
    print("💡 HuggingFace token gerekli olabilir: huggingface-cli login")
    sys.exit(1)

tokenizer.model_max_length = cfg["seq_len"]
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- dataset ----------
print(f"📂 Dataset yükleniyor: {args.dataset}")
ds = load_dataset("json", data_files=args.dataset, split="train")

if len(ds) == 0:
    print("❌ Dataset boş!")
    sys.exit(1)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=cfg["seq_len"],
    )

ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
print(f"✅ {len(ds)} örnek tokenize edildi")

# ---------- model ----------
print(f"🤖 Model yükleniyor: {args.model_name}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    print("💡 HuggingFace token gerekli olabilir: huggingface-cli login")
    sys.exit(1)

# RoPE scaling (opsiyonel)
if cfg.get("rope_scaling") is not None:
    model.config.rope_scaling = {
        "type": "linear",
        "factor": cfg["rope_scaling"]
    }

# ---------- batch hesap ----------
world_size = int(os.environ.get("WORLD_SIZE", 1))
tokens_per_step = cfg["tokens_per_step"]
per_device_bs = 1
grad_accum = max(1, tokens_per_step // (cfg["seq_len"] * world_size))

# ---------- training args ----------
out_dir = f"models/checkpoints/{args.run_id}"
os.makedirs(out_dir, exist_ok=True)

# YAML'dan gelen sayısal değerleri float'a çevir
peak_lr = float(cfg["peak_lr"])
warmup_ratio = float(cfg["warmup_ratio"])
weight_decay = float(cfg["weight_decay"])

print(f"🚀 Eğitim başlıyor...")
print(f"   Run ID: {args.run_id}")
print(f"   Model: {args.model_name}")
print(f"   Seq Len: {cfg['seq_len']}")
print(f"   Batch Size: {per_device_bs}")
print(f"   Grad Accum: {grad_accum}")
print(f"   Learning Rate: {peak_lr}")
print(f"   Output: {out_dir}")

training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=peak_lr,
    lr_scheduler_type=cfg["lr_schedule"],
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(out_dir)

print(f"✅ RUN {args.run_id} tamamlandı → {out_dir}")

