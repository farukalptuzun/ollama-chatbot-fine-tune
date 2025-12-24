print("🚀 train.py başlatılıyor...", flush=True)

import yaml
import argparse
import os
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
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
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

# ---------- seed ----------
seed = int(cfg.get("seed", 42))
set_seed(seed)

# ---------- tokenizer ----------
print(f"📥 Tokenizer yükleniyor: {args.model_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
except Exception as e:
    print(f"❌ Tokenizer yüklenemedi: {e}")
    print("💡 HuggingFace token gerekli olabilir: huggingface-cli login")
    sys.exit(1)

# Llama tokenizers genelde pad yok; eos'u pad yap
tokenizer.model_max_length = int(cfg["seq_len"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- dataset ----------
print(f"📂 Dataset yükleniyor: {args.dataset}")
ds = load_dataset("json", data_files=args.dataset, split="train")

if len(ds) == 0:
    print("❌ Dataset boş!")
    sys.exit(1)

# opsiyonel: debug/limit (runs.yaml içine max_train_samples koyarsan çalışır)
max_train_samples = cfg.get("max_train_samples", None)
if max_train_samples:
    max_train_samples = int(max_train_samples)
    ds = ds.select(range(min(len(ds), max_train_samples)))
    print(f"✅ max_train_samples aktif → {len(ds)} örneğe düşürüldü")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=int(cfg["seq_len"]),
    )

tokenized_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
print(f"✅ {len(tokenized_ds)} örnek tokenize edildi")

# Dataset split (train/eval)
print("📊 Dataset split ediliyor (train/eval)...")
split_ratio = float(cfg.get("eval_split_ratio", 0.01))
ds = tokenized_ds.train_test_split(test_size=split_ratio, seed=seed)
train_ds = ds["train"]
eval_ds = ds["test"]

# opsiyonel: eval'i sabitle (2M'de çok önemli)
max_eval_samples = int(cfg.get("max_eval_samples", 2000))
if max_eval_samples and len(eval_ds) > max_eval_samples:
    eval_ds = eval_ds.select(range(max_eval_samples))

print(f"✅ Train: {len(train_ds)}, Eval: {len(eval_ds)}")

# Token sayısını logla (debug için)
total_tokens = len(train_ds) * int(cfg["seq_len"])
print(f"🔢 Yaklaşık train token sayısı: {total_tokens:,}")

# ---------- model ----------
print(f"🤖 Model yükleniyor: {args.model_name}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,   # ✅ dtype yerine doğru parametre
        device_map="auto",
    )
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    print("💡 HuggingFace token gerekli olabilir: huggingface-cli login")
    sys.exit(1)

# RoPE scaling (opsiyonel)
if cfg.get("rope_scaling") is not None:
    model.config.rope_scaling = {"type": "linear", "factor": float(cfg["rope_scaling"])}

# gradient checkpointing ile uyum için
model.config.use_cache = False

# ---------- batch (Optuna uyumlu) ----------
per_device_bs = int(cfg.get("batch_size", 6))
grad_accum = int(cfg.get("grad_acc", 2))

# ---------- training args ----------
out_dir = f"models/checkpoints/{args.run_id}"
os.makedirs(out_dir, exist_ok=True)

learning_rate = float(cfg.get("learning_rate", cfg.get("peak_lr", 1.5e-5)))
warmup_ratio = float(cfg["warmup_ratio"])
weight_decay = float(cfg["weight_decay"])
num_epochs = int(cfg.get("num_epochs", 2))  # TrainingArguments int bekler

eval_steps = int(cfg.get("eval_steps", 2000))
save_steps = int(cfg.get("save_steps", 2000))
logging_steps = int(cfg.get("logging_steps", 50))
max_steps = cfg.get("max_steps", None)  # max_steps varsa epoch yerine kullanılır
if max_steps:
    max_steps = int(max_steps)

print("🚀 Eğitim başlıyor...")
print(f"   Run ID: {args.run_id}")
print(f"   Model: {args.model_name}")
print(f"   Seq Len: {cfg['seq_len']}")
print(f"   Batch Size: {per_device_bs}")
print(f"   Grad Accum: {grad_accum}")
print(f"   Effective Batch: {per_device_bs * grad_accum}")
print(f"   Learning Rate: {learning_rate}")
print(f"   Warmup Ratio: {warmup_ratio}")
print(f"   Weight Decay: {weight_decay}")
if max_steps:
    print(f"   Max Steps: {max_steps} (epoch override)")
else:
    print(f"   Epochs: {num_epochs}")
print(f"   Eval Steps: {eval_steps}")
print(f"   Save Steps: {save_steps}")
print(f"   Output: {out_dir}")

training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    lr_scheduler_type=cfg["lr_schedule"],
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    num_train_epochs=num_epochs if not max_steps else None,  # max_steps varsa epoch override edilir
    max_steps=max_steps,  # max_steps varsa epoch yerine kullanılır

    bf16=True,
    gradient_checkpointing=True,

    # ✅ hız: padding azalt
    group_by_length=True,

    logging_steps=logging_steps,

    eval_strategy="steps",  # Yeni transformers versiyonunda evaluation_strategy yerine eval_strategy
    eval_steps=eval_steps,

    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=int(cfg.get("save_total_limit", 2)),

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
)

# ✅ bf16/tensorcore için padding hizalaması
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Checkpoint kontrolü: varsa devam et, yoksa sıfırdan başla
checkpoint = get_last_checkpoint(out_dir)
if checkpoint:
    print(f"📂 Checkpoint bulundu, devam ediliyor: {checkpoint}")
else:
    print("🆕 Yeni eğitim başlatılıyor (checkpoint yok)")

trainer.train(resume_from_checkpoint=checkpoint)  # Checkpoint varsa devam eder, yoksa None = sıfırdan başlar
trainer.save_model(out_dir)

print(f"✅ RUN {args.run_id} tamamlandı → {out_dir}")
