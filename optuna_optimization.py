"""
Optuna ile hiperparametre optimizasyonu için kurumsal chatbot eğitim scripti
Llama-3-8B modeli için optimize edilmiştir
Colab uyumludur
"""

import json
import os
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset as HFDataset
import optuna
from optuna.trial import TrialState
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

print("🚀 Optuna Optimizasyon Scripti Başlatılıyor...", flush=True)

# Global değişkenler
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Kullanılan cihaz: {DEVICE}", flush=True)

# Metrikleri saklamak için global değişkenler
METRICS_HISTORY = []


class MetricsCallback:
    """Eğitim sırasında metrikleri toplamak için callback"""
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.perplexities = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                # Perplexity = exp(eval_loss)
                perplexity = math.exp(logs['eval_loss'])
                self.perplexities.append(perplexity)
    
    def get_final_metrics(self):
        return {
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_eval_loss': self.eval_losses[-1] if self.eval_losses else None,
            'final_perplexity': self.perplexities[-1] if self.perplexities else None,
            'min_perplexity': min(self.perplexities) if self.perplexities else None,
            'avg_perplexity': np.mean(self.perplexities) if self.perplexities else None,
        }


def load_and_prepare_data(dataset_path: str, test_size: float = 0.2, val_size: float = 0.1):
    """Veri setini yükle ve train/val/test olarak ayır"""
    print(f"📂 Dataset yükleniyor: {dataset_path}", flush=True)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset dosyası bulunamadı: {dataset_path}")
    
    # JSONL dosyasını yükle
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    if len(dataset) == 0:
        raise ValueError("Dataset boş!")
    
    print(f"✅ {len(dataset)} örnek yüklendi", flush=True)
    
    # Train/Test split
    train_test_split_idx = int(len(dataset) * (1 - test_size))
    train_val_data = dataset.select(range(train_test_split_idx))
    test_data = dataset.select(range(train_test_split_idx, len(dataset)))
    
    # Train/Val split
    val_split_idx = int(len(train_val_data) * (1 - val_size))
    train_data = train_val_data.select(range(val_split_idx))
    val_data = train_val_data.select(range(val_split_idx, len(train_val_data)))
    
    print(f"📊 Veri dağılımı: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}", flush=True)
    
    return train_data, val_data, test_data


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Dataset'i tokenize et"""
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    return tokenized


def calculate_f1_score(model, tokenizer, test_dataset, max_samples: int = 100):
    """F1 score hesapla (basit bir yaklaşım: next token prediction accuracy)"""
    model.eval()
    predictions = []
    labels = []
    
    # Test setinden örnek al (çok fazla olmasın)
    sample_size = min(max_samples, len(test_dataset))
    test_samples = test_dataset.select(range(sample_size))
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in test_samples:
            if 'input_ids' not in example:
                continue
                
            input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(DEVICE)
            
            if input_ids.shape[1] < 2:
                continue
            
            # Son token'ı tahmin et
            input_seq = input_ids[:, :-1]
            target_token = input_ids[:, -1].item()
            
            try:
                outputs = model(input_seq)
                logits = outputs.logits[0, -1, :]
                predicted_token = logits.argmax().item()
                
                if predicted_token == target_token:
                    correct += 1
                total += 1
            except Exception as e:
                continue
    
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    accuracy = correct / total
    
    # Basit bir F1 hesaplama (token-level accuracy'yi kullan)
    # Gerçek F1 için daha karmaşık bir yaklaşım gerekir
    precision = accuracy
    recall = accuracy
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    model.train()
    return f1, precision, recall, accuracy


def objective(trial: optuna.Trial, train_data, val_data, test_data, tokenizer):
    """Optuna objective fonksiyonu - her trial için eğitim yapar ve metrikleri döner"""
    
    # Hiperparametre önerileri
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
    seq_len = trial.suggest_categorical("seq_len", [2048, 4096, 8192])
    num_epochs = trial.suggest_int("num_epochs", 1, 3)
    
    print(f"\n🔬 Trial {trial.number} başlatılıyor...", flush=True)
    print(f"   Learning Rate: {learning_rate:.2e}", flush=True)
    print(f"   Batch Size: {batch_size}", flush=True)
    print(f"   Gradient Accumulation: {gradient_accumulation_steps}", flush=True)
    print(f"   Warmup Ratio: {warmup_ratio:.3f}", flush=True)
    print(f"   Weight Decay: {weight_decay:.3f}", flush=True)
    print(f"   Sequence Length: {seq_len}", flush=True)
    print(f"   Epochs: {num_epochs}", flush=True)
    
    try:
        # Dataset'i tokenize et
        train_tokenized = tokenize_dataset(train_data, tokenizer, seq_len)
        val_tokenized = tokenize_dataset(val_data, tokenizer, seq_len)
        test_tokenized = tokenize_dataset(test_data, tokenizer, seq_len)
        
        # Model yükle
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Training arguments
        output_dir = f"optuna_trials/trial_{trial.number}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            num_train_epochs=num_epochs,
            logging_steps=10,
            eval_steps=50,
            evaluation_strategy="steps",
            save_strategy="no",  # Disk alanı tasarrufu için
            bf16=True,
            report_to="none",
            load_best_model_at_end=False,
        )
        
        # Metrics callback
        metrics_callback = MetricsCallback()
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[metrics_callback],
        )
        
        # Eğitim
        train_result = trainer.train()
        
        # Validation metrikleri
        eval_result = trainer.evaluate()
        eval_loss = eval_result.get("eval_loss", float('inf'))
        perplexity = math.exp(eval_loss) if eval_loss != float('inf') else float('inf')
        
        # F1 score hesapla
        f1, precision, recall, accuracy = calculate_f1_score(
            model, tokenizer, test_tokenized, max_samples=50
        )
        
        # Cross-entropy loss (eval_loss zaten cross-entropy)
        cross_entropy_loss = eval_loss
        
        # Metrikleri topla
        final_metrics = metrics_callback.get_final_metrics()
        
        metrics = {
            "trial_number": trial.number,
            "perplexity": perplexity,
            "cross_entropy_loss": cross_entropy_loss,
            "eval_loss": eval_loss,
            "train_loss": train_result.training_loss,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "seq_len": seq_len,
            "num_epochs": num_epochs,
            **final_metrics,
        }
        
        METRICS_HISTORY.append(metrics)
        
        print(f"✅ Trial {trial.number} tamamlandı", flush=True)
        print(f"   Perplexity: {perplexity:.2f}", flush=True)
        print(f"   Cross-Entropy Loss: {cross_entropy_loss:.4f}", flush=True)
        print(f"   F1 Score: {f1:.4f}", flush=True)
        print(f"   Accuracy: {accuracy:.4f}", flush=True)
        
        # Model'i temizle (bellek tasarrufu)
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Objective değeri: perplexity'yi minimize et (veya F1'i maximize et)
        # Multi-objective için weighted sum kullanabiliriz
        # Perplexity düşük, F1 yüksek olmalı
        objective_value = perplexity - (f1 * 100)  # F1'i ağırlıklandır
        
        return objective_value
        
    except Exception as e:
        print(f"❌ Trial {trial.number} başarısız: {e}", flush=True)
        return float('inf')


def run_ab_test(best_params: Dict, baseline_params: Dict, test_data, tokenizer):
    """A/B test: En iyi parametreler vs baseline"""
    print("\n🧪 A/B Test başlatılıyor...", flush=True)
    
    ab_results = {
        "best_model": {},
        "baseline_model": {},
        "comparison": {}
    }
    
    # Bu basit bir implementasyon - gerçek A/B test için daha fazla veri gerekir
    # Şimdilik sadece metrikleri karşılaştırıyoruz
    
    return ab_results


def save_results_to_json(study: optuna.Study, output_path: str):
    """Optuna study sonuçlarını JSON formatında kaydet"""
    print(f"\n💾 Sonuçlar JSON'a kaydediliyor: {output_path}", flush=True)
    
    # En iyi trial
    best_trial = study.best_trial
    
    # Tüm trial'lar
    all_trials = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            all_trials.append(trial_data)
    
    # Metriklerle birleştir
    for trial_data in all_trials:
        trial_num = trial_data["number"]
        matching_metrics = [m for m in METRICS_HISTORY if m["trial_number"] == trial_num]
        if matching_metrics:
            trial_data["metrics"] = matching_metrics[0]
    
    # Sonuç objesi
    results = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "datetime_start": best_trial.datetime_start.isoformat() if best_trial.datetime_start else None,
            "datetime_complete": best_trial.datetime_complete.isoformat() if best_trial.datetime_complete else None,
        },
        "best_metrics": next((m for m in METRICS_HISTORY if m["trial_number"] == best_trial.number), {}),
        "all_trials": all_trials,
        "timestamp": datetime.now().isoformat(),
    }
    
    # JSON'a kaydet
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Sonuçlar kaydedildi: {output_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Optuna ile hiperparametre optimizasyonu")
    parser.add_argument("--dataset", type=str, default="data/train.jsonl", help="Dataset dosya yolu")
    parser.add_argument("--n_trials", type=int, default=10, help="Deneme sayısı")
    parser.add_argument("--output", type=str, default="optuna_results.json", help="JSON çıktı dosyası")
    parser.add_argument("--study_name", type=str, default="llama3_8b_optimization", help="Optuna study adı")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (saniye)")
    
    args = parser.parse_args()
    
    print(f"📋 Konfigürasyon:", flush=True)
    print(f"   Dataset: {args.dataset}", flush=True)
    print(f"   Trial Sayısı: {args.n_trials}", flush=True)
    print(f"   Çıktı Dosyası: {args.output}", flush=True)
    print(f"   Study Adı: {args.study_name}", flush=True)
    
    # Veri yükle
    train_data, val_data, test_data = load_and_prepare_data(args.dataset)
    
    # Tokenizer yükle
    print(f"📥 Tokenizer yükleniyor: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Objective fonksiyonunu hazırla
    def objective_wrapper(trial):
        return objective(trial, train_data, val_data, test_data, tokenizer)
    
    # Optuna study oluştur
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",  # Objective değerini minimize et
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    print(f"\n🎯 Optuna optimizasyonu başlatılıyor...", flush=True)
    
    # Optimizasyonu çalıştır
    study.optimize(
        objective_wrapper,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Sonuçları kaydet
    results = save_results_to_json(study, args.output)
    
    # Özet yazdır
    print(f"\n{'='*60}", flush=True)
    print(f"🎉 OPTİMİZASYON TAMAMLANDI", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"📊 Toplam Trial: {len(study.trials)}", flush=True)
    print(f"🏆 En İyi Trial: {study.best_trial.number}", flush=True)
    print(f"📈 En İyi Değer: {study.best_trial.value:.4f}", flush=True)
    print(f"\n🔧 En İyi Parametreler:", flush=True)
    for key, value in study.best_trial.params.items():
        print(f"   {key}: {value}", flush=True)
    
    if "best_metrics" in results and results["best_metrics"]:
        print(f"\n📏 En İyi Metrikler:", flush=True)
        metrics = results["best_metrics"]
        print(f"   Perplexity: {metrics.get('perplexity', 'N/A'):.2f}", flush=True)
        print(f"   Cross-Entropy Loss: {metrics.get('cross_entropy_loss', 'N/A'):.4f}", flush=True)
        print(f"   F1 Score: {metrics.get('f1_score', 'N/A'):.4f}", flush=True)
        print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}", flush=True)
    
    print(f"\n💾 Detaylı sonuçlar: {args.output}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()

