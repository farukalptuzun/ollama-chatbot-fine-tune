# 🚀 Ollama Chatbot Fine-Tune

Kurumsal chatbot geliştirme için **decoder-only** dil modellerini (Llama-3-8B, Qwen, vb.) fine-tune etmek üzere tasarlanmış, esnek ve ölçeklenebilir bir eğitim framework'ü.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım](#-kullanım)
- [Deney Matrisi Sistemi](#-deney-matrisi-sistemi)
- [Optuna ile Hiperparametre Optimizasyonu](#-optuna-ile-hiperparametre-optimizasyonu)
- [Dataset Formatı](#-dataset-formatı)
- [Checkpoint Yönetimi](#-checkpoint-yönetimi)
- [Sorun Giderme](#-sorun-giderme)

## ✨ Özellikler

### 🎯 Ana Özellikler

- ✅ **Deney Matrisi Sistemi**: YAML tabanlı konfigürasyon ile kolay deney yönetimi
- ✅ **Otomatik Checkpoint Yönetimi**: Her deney için ayrı checkpoint dizini
- ✅ **HuggingFace Entegrasyonu**: Tüm decoder-only modelleri destekler (Llama, Qwen, Mistral, vb.)
- ✅ **RoPE Scaling Desteği**: Uzun sequence length'ler için RoPE scaling
- ✅ **Gradient Accumulation**: Büyük batch size'ları için otomatik hesaplama
- ✅ **BF16 Training**: Bellek verimli mixed precision training
- ✅ **Error Handling**: Kapsamlı hata kontrolü ve açıklayıcı mesajlar

### 🔬 Optuna Optimizasyon

- ✅ **Otomatik Hiperparametre Optimizasyonu**: TPE (Tree-structured Parzen Estimator) algoritması
- ✅ **Kapsamlı Metrikler**: Perplexity, Cross-Entropy Loss, F1 Score, Accuracy
- ✅ **A/B Test Desteği**: Farklı konfigürasyonları karşılaştırma
- ✅ **JSON Çıktı**: Tüm sonuçlar JSON formatında kaydedilir
- ✅ **Colab Uyumlu**: Google Colab'de çalışmaya hazır

## 📁 Proje Yapısı

```
ollama-chatbot-fine-tune/
├── train.py                    # Ana eğitim script'i
├── runs.yaml                   # Deney matrisi konfigürasyonu
├── run_experiment.sh           # Deney çalıştırma script'i
├── optuna_optimization.py       # Optuna hiperparametre optimizasyonu
├── setup.sh                    # Otomatik kurulum script'i
├── requirements.txt            # Python bağımlılıkları
├── colab_optuna_training.ipynb # Google Colab notebook
├── OPTUNA_README.md            # Optuna kullanım kılavuzu
├── data/
│   └── train.jsonl            # Eğitim dataset'i (JSONL format)
└── models/
    └── checkpoints/
        └── E01/               # Deney checkpoint'leri
            ├── config.json
            ├── model.safetensors
            └── tokenizer.json
```

## 🛠️ Kurulum

### Gereksinimler

- Python 3.10+
- CUDA uyumlu GPU (önerilir, en az 16GB VRAM)
- Git

### Otomatik Kurulum

```bash
# Projeyi klonlayın
git clone https://github.com/farukalptuzun/ollama-chatbot-fine-tune.git
cd ollama-chatbot-fine-tune

# Otomatik kurulum script'ini çalıştırın
bash setup.sh
```

### Manuel Kurulum

```bash
# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows

# Bağımlılıkları yükle
pip install --upgrade pip
pip install -r requirements.txt

# Gerekli dizinleri oluştur
mkdir -p models/checkpoints data
```

### HuggingFace Token Ayarlama

Gated modeller (Llama-3-8B gibi) için HuggingFace token gerekir:

```bash
huggingface-cli login
# veya
hf auth login
```

Token'ı oluşturmak için: https://huggingface.co/settings/tokens

## 🚀 Hızlı Başlangıç

### 1. Dataset Hazırlama

Dataset'iniz JSONL formatında olmalı:

```bash
# data/train.jsonl örneği
{"text": "Müşteri: Merhaba, ürününüz hakkında bilgi alabilir miyim?\nAsistan: Tabii ki! Size nasıl yardımcı olabilirim?"}
{"text": "Müşteri: Fiyat bilgisi almak istiyorum.\nAsistan: Hangi ürün için fiyat bilgisi istiyorsunuz?"}
```

### 2. Deney Konfigürasyonu

`runs.yaml` dosyasına yeni deney ekleyin:

```yaml
E01:
  tokenizer: unigram
  vocab_size: 64000
  seq_len: 4096
  lr_schedule: cosine
  peak_lr: 3e-4
  warmup_ratio: 0.02
  weight_decay: 0.1
  rope_scaling: null
  tokens_per_step: 1048576
```

### 3. Eğitimi Başlatma

```bash
# Venv'i aktifleştir
source venv/bin/activate

# Deneyi çalıştır
./run_experiment.sh E01
```

## 📖 Kullanım

### Temel Kullanım

```bash
# Deney ID'si ile çalıştır
./run_experiment.sh E01

# Farklı model ile
python3 train.py \
  --run_id E01 \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset data/train.jsonl
```

### Parametreler

#### `train.py` Parametreleri

- `--run_id` (required): `runs.yaml`'daki deney ID'si
- `--model_name` (default: `meta-llama/Meta-Llama-3-8B`): HuggingFace model adı veya yerel path
- `--dataset` (default: `data/train.jsonl`): Dataset dosya yolu

#### `runs.yaml` Konfigürasyon Parametreleri

- `tokenizer`: Tokenizer tipi (unigram, bpe) - şu an kullanılmıyor, gelecekte kullanılacak
- `vocab_size`: Vocabulary boyutu - şu an kullanılmıyor
- `seq_len`: Maksimum sequence length (2048, 4096, 8192, vb.)
- `lr_schedule`: Learning rate scheduler tipi (`cosine`, `linear`, `constant`)
- `peak_lr`: Peak learning rate (örn: `3e-4`, `1e-4`)
- `warmup_ratio`: Warmup oranı (0.0-1.0 arası, örn: `0.02`)
- `weight_decay`: Weight decay değeri (örn: `0.1`)
- `rope_scaling`: RoPE scaling faktörü (`null` veya sayısal değer)
- `tokens_per_step`: Her step'te işlenecek token sayısı (batch size hesaplama için)

## 🧪 Deney Matrisi Sistemi

### Deney Ekleme

`runs.yaml` dosyasına yeni deney ekleyerek farklı konfigürasyonları test edebilirsiniz:

```yaml
E01:
  tokenizer: unigram
  vocab_size: 64000
  seq_len: 4096
  lr_schedule: cosine
  peak_lr: 3e-4
  warmup_ratio: 0.02
  weight_decay: 0.1
  rope_scaling: null
  tokens_per_step: 1048576

E02:
  tokenizer: unigram
  vocab_size: 64000
  seq_len: 8192          # Daha uzun sequence
  lr_schedule: cosine
  peak_lr: 3e-4
  warmup_ratio: 0.02
  weight_decay: 0.1
  rope_scaling: null
  tokens_per_step: 1048576

E03:
  tokenizer: bpe
  vocab_size: 64000
  seq_len: 8192
  lr_schedule: cosine
  peak_lr: 1e-4         # Daha düşük learning rate
  warmup_ratio: 0.05    # Daha fazla warmup
  weight_decay: 0.1
  rope_scaling: 2.0     # RoPE scaling aktif
  tokens_per_step: 1048576
```

### Batch Size Hesaplama

Sistem otomatik olarak batch size'ı hesaplar:

```
gradient_accumulation_steps = tokens_per_step / (seq_len * world_size)
per_device_batch_size = 1 (sabit)
```

Örnek:
- `tokens_per_step = 1048576`
- `seq_len = 4096`
- `world_size = 1` (tek GPU)
- `gradient_accumulation_steps = 1048576 / (4096 * 1) = 256`

## 🔬 Optuna ile Hiperparametre Optimizasyonu

### Kullanım

```bash
python3 optuna_optimization.py \
    --dataset data/train.jsonl \
    --n_trials 20 \
    --output optuna_results.json \
    --study_name llama3_8b_chatbot_optimization
```

### Optimize Edilen Parametreler

- **learning_rate**: 1e-5 ile 1e-3 arası (log scale)
- **batch_size**: 1, 2, 4
- **gradient_accumulation_steps**: 1 ile 8 arası
- **warmup_ratio**: 0.01 ile 0.1 arası
- **weight_decay**: 0.01 ile 0.3 arası
- **seq_len**: 2048, 4096, 8192
- **num_epochs**: 1 ile 3 arası

### Sonuçları İnceleme

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

Detaylı bilgi için: [OPTUNA_README.md](OPTUNA_README.md)

## 📊 Dataset Formatı

### JSONL Formatı

Her satır bir JSON objesi olmalı ve `text` alanı içermeli:

```json
{"text": "Türkiye bir ülkedir. Başkenti Ankara'dır."}
{"text": "Transformer models use self-attention mechanisms."}
{"text": "Python programlama dili veri bilimi alanında yaygın kullanılır."}
```

### Chatbot Formatı (Önerilen)

```json
{"text": "Müşteri: Merhaba, ürününüz hakkında bilgi alabilir miyim?\nAsistan: Tabii ki! Size nasıl yardımcı olabilirim?\nMüşteri: Fiyat bilgisi almak istiyorum.\nAsistan: Hangi ürün için fiyat bilgisi istiyorsunuz?"}
```

### Dataset Hazırlama İpuçları

1. **Minimum Dataset Boyutu**: En az 1000 örnek önerilir
2. **Kalite > Miktar**: Az ama kaliteli veri, çok ama düşük kaliteli veriden daha iyidir
3. **Dengeli Dağılım**: Farklı konu ve senaryoları kapsamalı
4. **Temizlik**: Gereksiz karakterler, HTML tag'leri temizlenmeli

## 💾 Checkpoint Yönetimi

### Checkpoint Konumu

Her deney için checkpoint'ler ayrı dizinde saklanır:

```
models/checkpoints/E01/
├── config.json
├── generation_config.json
├── model.safetensors (veya model-*.safetensors)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Checkpoint Kullanma

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Checkpoint'ten model yükle
model = AutoModelForCausalLM.from_pretrained("models/checkpoints/E01")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoints/E01")

# Inference
inputs = tokenizer("Merhaba, nasılsın?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Checkpoint Yedekleme

```bash
# Checkpoint'i yedekle
tar -czf E01_checkpoint.tar.gz models/checkpoints/E01/

# Yedekten geri yükle
tar -xzf E01_checkpoint.tar.gz
```

## 🐛 Sorun Giderme

### HuggingFace Token Hatası

```bash
# Token'ı kontrol et
hf auth whoami

# Yeni token ile giriş yap
huggingface-cli login
```

### CUDA Out of Memory

1. **Batch Size Azalt**: `tokens_per_step` değerini azaltın
2. **Sequence Length Azalt**: `seq_len` değerini düşürün (4096 → 2048)
3. **Gradient Checkpointing**: Model'e `gradient_checkpointing=True` ekleyin
4. **CPU Offloading**: `device_map="auto"` zaten aktif, daha fazla offload için `accelerate` kullanın

### Dataset Format Hatası

```bash
# Dataset'i kontrol et
head -n 5 data/train.jsonl

# Her satırın geçerli JSON olduğundan emin ol
python3 -c "import json; [print(json.loads(line)) for line in open('data/train.jsonl')]"
```

### Import Hatası (sympy yavaş yükleniyor)

İlk import uzun sürebilir (1-2 dakika). Bu normaldir. Sonraki çalıştırmalarda hızlı olacaktır.

```bash
# Cache'i önceden oluştur
python3 -c "import sympy; print('sympy cache oluşturuldu')"
```

### Python Komutu Bulunamadı

```bash
# Python3 kullan
python3 train.py --run_id E01

# Veya venv içindeki Python'u kullan
source venv/bin/activate
python train.py --run_id E01
```

## 📈 Performans İpuçları

### GPU Bellek Optimizasyonu

1. **BF16 Training**: Zaten aktif (`bf16=True`)
2. **Gradient Accumulation**: Büyük batch size'lar için kullanılır
3. **CPU Offloading**: `device_map="auto"` ile otomatik
4. **Gradient Checkpointing**: Daha fazla bellek tasarrufu için

### Eğitim Hızlandırma

1. **Mixed Precision**: BF16 zaten aktif
2. **DataLoader Workers**: `num_workers` parametresi eklenebilir
3. **Compile Model**: PyTorch 2.0+ için `model = torch.compile(model)`

## 🔄 Sonraki Adımlar

1. **Optuna ile Optimizasyon**: En iyi hiperparametreleri bulun
2. **Final Eğitim**: En iyi parametrelerle tüm dataset ile eğitin
3. **Model Değerlendirme**: Test seti ile modeli değerlendirin
4. **Deployment**: Modeli production'a alın (Ollama, vLLM, vb.)

## 📚 Kaynaklar

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👤 Yazar

**Faruk Alptüzün**

- GitHub: [@farukalptuzun](https://github.com/farukalptuzun)

## 🙏 Teşekkürler

- HuggingFace ekibine Transformers kütüphanesi için
- Meta AI'ya Llama modelleri için
- Optuna ekibine hiperparametre optimizasyonu için

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

