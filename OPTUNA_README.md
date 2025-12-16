# 🚀 Optuna ile Llama-3-8B Hiperparametre Optimizasyonu

Bu sistem, kurumsal chatbot için Llama-3-8B modelini fine-tune ederken Optuna ile otomatik hiperparametre optimizasyonu yapar.

## 📋 Özellikler

- ✅ **Optuna ile Otomatik Optimizasyon**: TPE (Tree-structured Parzen Estimator) algoritması ile en iyi hiperparametreleri bulur
- ✅ **Kapsamlı Metrikler**: 
  - Perplexity
  - Cross-Entropy Loss
  - F1 Score
  - Precision, Recall, Accuracy
- ✅ **A/B Test Desteği**: Farklı konfigürasyonları karşılaştırma
- ✅ **JSON Çıktı**: Tüm sonuçlar JSON formatında kaydedilir
- ✅ **Colab Uyumlu**: Google Colab'de çalışmaya hazır

## 📦 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

Veya manuel olarak:

```bash
pip install torch transformers datasets accelerate optuna scikit-learn pyyaml sentencepiece numpy pandas
```

### HuggingFace Token

Llama-3-8B modelini kullanmak için HuggingFace token'ına ihtiyacınız var:

```bash
huggingface-cli login
```

## 🚀 Kullanım

### 1. Dataset Hazırlama

Dataset'iniz JSONL formatında olmalı ve her satırda bir JSON objesi bulunmalı:

```json
{"text": "Müşteri: Merhaba, ürününüz hakkında bilgi alabilir miyim? Asistan: Tabii ki! Size nasıl yardımcı olabilirim?"}
{"text": "Müşteri: Fiyat bilgisi almak istiyorum. Asistan: Hangi ürün için fiyat bilgisi istiyorsunuz?"}
```

### 2. Komut Satırından Çalıştırma

```bash
python optuna_optimization.py \
    --dataset data/train.jsonl \
    --n_trials 20 \
    --output optuna_results.json \
    --study_name llama3_8b_chatbot_optimization
```

### Parametreler

- `--dataset`: Dataset dosya yolu (varsayılan: `data/train.jsonl`)
- `--n_trials`: Deneme sayısı (varsayılan: 10)
- `--output`: JSON çıktı dosyası (varsayılan: `optuna_results.json`)
- `--study_name`: Optuna study adı (varsayılan: `llama3_8b_optimization`)
- `--timeout`: Timeout süresi saniye cinsinden (opsiyonel)

### 3. Google Colab'de Kullanım

1. `colab_optuna_training.ipynb` dosyasını Colab'e yükleyin
2. Dataset'inizi yükleyin veya örnek dataset'i kullanın
3. `optuna_optimization.py` dosyasını Colab'e yükleyin
4. Notebook'u hücre hücre çalıştırın

## 📊 Optimize Edilen Hiperparametreler

Script aşağıdaki hiperparametreleri optimize eder:

- **learning_rate**: 1e-5 ile 1e-3 arası (log scale)
- **batch_size**: 1, 2, 4
- **gradient_accumulation_steps**: 1 ile 8 arası
- **warmup_ratio**: 0.01 ile 0.1 arası
- **weight_decay**: 0.01 ile 0.3 arası
- **seq_len**: 2048, 4096, 8192
- **num_epochs**: 1 ile 3 arası

## 📈 Çıktı Formatı

JSON çıktı dosyası şu yapıda olacaktır:

```json
{
  "study_name": "llama3_8b_chatbot_optimization",
  "n_trials": 20,
  "best_trial": {
    "number": 5,
    "value": 12.34,
    "params": {
      "learning_rate": 2e-4,
      "batch_size": 2,
      "gradient_accumulation_steps": 4,
      "warmup_ratio": 0.05,
      "weight_decay": 0.1,
      "seq_len": 4096,
      "num_epochs": 2
    }
  },
  "best_metrics": {
    "perplexity": 15.23,
    "cross_entropy_loss": 2.72,
    "f1_score": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "accuracy": 0.85
  },
  "all_trials": [...]
}
```

## 🔍 Metrikler Açıklaması

### Perplexity
Modelin tahmin belirsizliğini ölçer. Düşük değer daha iyidir. `exp(eval_loss)` formülü ile hesaplanır.

### Cross-Entropy Loss
Modelin tahmin hatasını ölçer. Düşük değer daha iyidir. Validation loss olarak hesaplanır.

### F1 Score
Precision ve Recall'un harmonik ortalaması. Yüksek değer daha iyidir (0-1 arası).

### Accuracy
Doğru tahmin yüzdesi. Yüksek değer daha iyidir (0-1 arası).

## 💡 İpuçları

1. **Trial Sayısı**: Daha iyi sonuçlar için en az 20-30 trial önerilir
2. **GPU Belleği**: Llama-3-8B için en az 16GB GPU belleği önerilir
3. **Dataset Boyutu**: En az 1000 örnek önerilir, daha fazla daha iyi
4. **Zaman**: Her trial 30-60 dakika sürebilir (GPU'ya bağlı)

## 🐛 Sorun Giderme

### CUDA Out of Memory
- Batch size'ı azaltın (script otomatik olarak 1, 2, 4 dener)
- Gradient accumulation steps'i artırın
- Sequence length'i azaltın

### HuggingFace Token Hatası
```bash
huggingface-cli login
```

### Dataset Format Hatası
Dataset'iniz JSONL formatında olmalı ve her satırda `{"text": "..."}` formatında olmalı.

## 📝 Örnek Kullanım

```python
import json

# Sonuçları yükle
with open("optuna_results.json", "r") as f:
    results = json.load(f)

# En iyi parametreleri al
best_params = results["best_trial"]["params"]
print(f"En iyi learning rate: {best_params['learning_rate']}")
print(f"En iyi batch size: {best_params['batch_size']}")

# En iyi metrikleri görüntüle
best_metrics = results["best_metrics"]
print(f"En iyi perplexity: {best_metrics['perplexity']:.2f}")
print(f"En iyi F1 score: {best_metrics['f1_score']:.4f}")
```

## 🔄 Sonraki Adımlar

En iyi parametreleri bulduktan sonra:

1. `train.py` scriptinizi en iyi parametrelerle güncelleyin
2. Final modeli tüm dataset ile eğitin
3. Modeli test edin ve deploy edin

## 📞 Destek

Sorun yaşarsanız:
- Script loglarını kontrol edin
- GPU bellek kullanımını izleyin
- Dataset formatını doğrulayın

