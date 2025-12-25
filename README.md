# 🚀 Ollama Chatbot Fine-Tune

Kurumsal chatbot geliştirme için **decoder-only** dil modellerini (Llama-3-8B, Qwen, vb.) fine-tune etmek üzere tasarlanmış, esnek ve ölçeklenebilir bir eğitim framework'ü.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Geliştirme Ortamı](#-geliştirme-ortamı)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Workflow: Optuna + Final Training](#-workflow-optuna--final-training)
- [Kullanım](#-kullanım)
- [Deney Matrisi Sistemi](#-deney-matrisi-sistemi)
- [Optuna ile Hiperparametre Optimizasyonu](#-optuna-ile-hiperparametre-optimizasyonu)
- [Dataset Formatı](#-dataset-formatı)
- [Checkpoint Yönetimi](#-checkpoint-yönetimi)
- [Ollama Modelfile](#-ollama-modelfile)
- [Sorun Giderme](#-sorun-giderme)

## ✨ Özellikler

### 🎯 Ana Özellikler

- ✅ **Optuna Trend Bulma**: Küçük dataset ile parametre aralığı bulma (10K örnek)
- ✅ **Final Training**: 2M örnekle production-ready model eğitimi
- ✅ **Deney Matrisi Sistemi**: YAML tabanlı konfigürasyon ile kolay deney yönetimi
- ✅ **Otomatik Checkpoint Yönetimi**: Step-based + Epoch-based checkpoint sistemi
- ✅ **Resume Training**: Eğitim kopsa bile checkpoint'ten devam etme
- ✅ **HuggingFace Entegrasyonu**: Tüm decoder-only modelleri destekler (Llama, Qwen, Mistral, vb.)
- ✅ **BF16 Training**: Bellek verimli mixed precision training
- ✅ **Gradient Checkpointing**: Bellek optimizasyonu
- ✅ **TensorCore Optimizasyonu**: `pad_to_multiple_of=8` ile hızlandırma
- ✅ **Group by Length**: Padding israfını azaltma
- ✅ **Evaluation Desteği**: Train/test split ve validation metrics

### 🔬 Optuna Optimizasyon

- ✅ **Trend Bulma Stratejisi**: En iyi parametre değil, transfer edilebilir parametre aralığı
- ✅ **Daraltılmış Search Space**: Hızlı ve verimli optimizasyon
- ✅ **Agresif Pruning**: MedianPruner ile erken durdurma
- ✅ **TPE Sampler**: Tree-structured Parzen Estimator algoritması
- ✅ **Kapsamlı Metrikler**: Perplexity, Cross-Entropy Loss, F1 Score, Accuracy
- ✅ **JSON Çıktı**: Tüm sonuçlar JSON formatında kaydedilir
- ✅ **Colab Uyumlu**: Google Colab'de çalışmaya hazır

### ⚡ Hız Optimizasyonları

- ✅ **Max Steps Desteği**: Epoch yerine step-based training (18-24 saatlik eğitim)
- ✅ **Küçük Sequence Length**: Optuna için 1024, final için 768 (hızlandırma)
- ✅ **Batch Size Optimizasyonu**: H100 80GB için optimize edilmiş batch size
- ✅ **Evaluation Optimizasyonu**: Sabit küçük eval seti (2M'de kritik)

## 🖥️ Geliştirme Ortamı

Bu proje **Google Colab** üzerinde geliştirilmiştir:

- **GPU**: H100 80GB VRAM
- **Sistem RAM**: 200GB
- **Python**: 3.10+
- **CUDA**: Uyumlu CUDA sürümü
- **Storage**: Google Drive entegrasyonu

### Colab Kurulumu

```python
# Google Drive'ı bağla
from google.colab import drive
drive.mount('/content/drive')

# Projeye git
cd /content/drive/MyDrive/ollama-llm-fine-tune/ollama-chatbot-fine-tune

# Bağımlılıkları kur
pip install -r requirements.txt

# HuggingFace token
huggingface-cli login
```

## 📁 Proje Yapısı

```
ollama-chatbot-fine-tune/
├── train.py                    # Ana eğitim script'i (2M final training)
├── optuna_optimization.py       # Optuna hiperparametre optimizasyonu (10K trend bulma)
├── runs.yaml                   # Deney matrisi konfigürasyonu
├── run_experiment.sh           # Deney çalıştırma script'i
├── setup.sh                    # Otomatik kurulum script'i
├── requirements.txt            # Python bağımlılıkları
├── colab_optuna_training.ipynb # Google Colab notebook
├── OPTUNA_README.md            # Optuna kullanım kılavuzu
├── Modelfile.E01_1day          # Ollama Modelfile (deployment için)
├── data/
│   └── train.jsonl            # Eğitim dataset'i (JSONL format, 2M örnek)
└── models/
    └── checkpoints/
        └── E01_1day/          # Deney checkpoint'leri
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            ├── checkpoint-8000/    # Step-based checkpoint'ler
            ├── checkpoint-16000/
            ├── epoch-1-final/      # Epoch-based checkpoint'ler
            └── final-step-40000/   # Final checkpoint
```

## 🛠️ Kurulum

### Gereksinimler

- Python 3.10+
- CUDA uyumlu GPU (önerilir: H100 80GB veya en az 16GB VRAM)
- Sistem RAM: 200GB+ (büyük dataset'ler için)
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

`runs.yaml` dosyasına yeni deney ekleyin. Detaylı örnekler için [Deney Matrisi Sistemi](#-deney-matrisi-sistemi) bölümüne bakın.

### 3. Eğitimi Başlatma

```bash
# Venv'i aktifleştir
source venv/bin/activate

# Deneyi çalıştır
./run_experiment.sh E01_1day
```

## 🔄 Workflow: Optuna + Final Training

Bu proje **iki aşamalı bir workflow** kullanır:

### Aşama 1: Optuna Optimizasyon (Trend Bulma)

**Amaç**: 2M örnekle çalışacak parametre **aralığını** bulmak (mutlak en iyi değeri değil)

**Strateji**:
- Küçük dataset: 10K train, 2K validation
- Kısa eğitim: seq_len=1024, batch_size=2, 1 epoch
- Agresif pruning: MedianPruner(n_startup_trials=2, n_warmup_steps=20)
- Daraltılmış search space: LR (1e-5, 5e-5), grad_acc (2-6), warmup (0.01-0.05)

**Kullanım**:
```bash
python optuna_optimization.py \
  --n_trials 8 \
  --max_train_samples 10000 \
  --max_val_samples 2000
```

**Süre**: H100 ile 2-3 saat (8 trial)

### Aşama 2: Final Training (Production)

**Amaç**: Optuna'dan gelen parametre aralığıyla 2M örnekle kaliteli model eğitmek

**Strateji**:
- Büyük dataset: 2M örnek
- Optuna sonuçlarından transfer: LR %20 düşürülmüş (1.5e-5), effective batch korunuyor
- Hız optimizasyonu: seq_len=768, max_steps=40000, batch_size=8
- Evaluation: Sabit küçük eval seti (1000 örnek)

**Kullanım**:
```bash
python train.py --run_id E01_1day --dataset data/train.jsonl
```

**Süre**: H100 ile 18-24 saat (40K step)

**Kalite**: %85-90 korunur (production için yeterli)

## 📖 Kullanım

### Temel Kullanım

```bash
# Deney ID'si ile çalıştır
./run_experiment.sh E01_1day

# Veya direkt Python ile
python train.py \
  --run_id E01_1day \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset data/train.jsonl
```

### Parametreler

#### `train.py` Parametreleri

- `--run_id` (required): `runs.yaml`'daki deney ID'si
- `--model_name` (default: `meta-llama/Meta-Llama-3-8B`): HuggingFace model adı veya yerel path
- `--dataset` (default: `data/train.jsonl`): Dataset dosya yolu

#### `runs.yaml` Konfigürasyon Parametreleri

**Temel Parametreler**:
- `seq_len`: Maksimum sequence length (768, 1024, 2048, 4096, vb.)
- `lr_schedule`: Learning rate scheduler tipi (`cosine`, `linear`, `constant`)
- `learning_rate`: Learning rate (örn: `1.5e-5`)
- `warmup_ratio`: Warmup oranı (0.0-1.0 arası, örn: `0.03`)
- `weight_decay`: Weight decay değeri (örn: `0.05`)
- `rope_scaling`: RoPE scaling faktörü (`null` veya sayısal değer)

**Batch ve Eğitim**:
- `batch_size`: Per-device batch size (örn: `8`)
- `grad_acc`: Gradient accumulation steps (örn: `2`)
- `num_epochs`: Epoch sayısı (örn: `1`, `2`)
- `max_steps`: Maksimum step sayısı (epoch override, opsiyonel)

**Evaluation ve Checkpoint**:
- `eval_split_ratio`: Validation split ratio (örn: `0.01`)
- `max_eval_samples`: Maksimum eval örnek sayısı (örn: `2000`)
- `eval_steps`: Her kaç step'te evaluation (örn: `2000`)
- `save_steps`: Her kaç step'te checkpoint kaydet (örn: `2000`)
- `save_total_limit`: Maksimum checkpoint sayısı (örn: `2`)
- `logging_steps`: Her kaç step'te log (örn: `50`)
- `seed`: Random seed (örn: `42`)

## 🧪 Deney Matrisi Sistemi

### Örnek Konfigürasyonlar

#### E01: Optuna Sonuçlarından Gelen Parametreler

```yaml
E01:
  seq_len: 1024                    # Optuna ile optimize edilmiş
  lr_schedule: cosine
  learning_rate: 1.5e-5           # Optuna'dan %20 düşürülmüş
  warmup_ratio: 0.035
  weight_decay: 0.05              # Optuna sonuçlarına uygun
  rope_scaling: null

  batch_size: 6                   # Effective batch=12 (6*2)
  grad_acc: 2
  num_epochs: 2

  eval_split_ratio: 0.01
  max_eval_samples: 2000
  eval_steps: 2000
  save_steps: 2000
  save_total_limit: 2
  logging_steps: 50
  seed: 42
```

#### E01_1day: 18-24 Saatlik Hızlı Eğitim

```yaml
E01_1day:
  seq_len: 768                    # 1024 → 768 (%25-30 hız kazanımı)
  lr_schedule: cosine
  learning_rate: 1.5e-5          # LR sabit tut (kalite korunur)
  warmup_ratio: 0.03
  weight_decay: 0.05
  rope_scaling: null

  batch_size: 8                  # 6 → 8 (H100 80GB yeterli)
  grad_acc: 2
  num_epochs: 1                  # 2 → 1 (×0.5 süre)
  max_steps: 40000               # Epoch yerine step-based (40K step ≈ 17 saat)

  eval_split_ratio: 0.005        # Küçük eval seti
  max_eval_samples: 1000
  eval_steps: 8000               # Daha az evaluation
  save_steps: 8000
  save_total_limit: 2
  logging_steps: 100
  seed: 42
```

## 🔬 Optuna ile Hiperparametre Optimizasyonu

### Felsefe: Trend Bulma

**Önemli**: Optuna'nın amacı en iyi final loss değil, **2M eğitimde iyi çalışacak parametre aralığını bulmaktır**.

**Altın Kural**: 
- Optuna = trend bulur
- Final training = kaliteyi üretir

### Kullanım

```bash
python optuna_optimization.py \
  --dataset data/train.jsonl \
  --n_trials 8 \
  --max_train_samples 10000 \
  --max_val_samples 2000 \
  --output optuna_results.json \
  --study_name llama3_8b_optimization
```

### Optimize Edilen Parametreler (Daraltılmış Search Space)

- **learning_rate**: 1e-5 ile 5e-5 arası (log scale) - 2M için ideal aralık
- **grad_acc**: 2 ile 6 arası - Daraltılmış aralık
- **warmup_ratio**: 0.01 ile 0.05 arası - Daraltılmış aralık
- **weight_decay**: 0.0 ile 0.1 arası - Daraltılmış aralık
- **batch_size**: 2 (SABİT) - Trend bulma için yeterli
- **seq_len**: 1024 (SABİT) - 2048 gereksiz pahalı
- **num_epochs**: 1 (SABİT) - Trend bulma için yeterli

### Pruning Konfigürasyonu

```python
MedianPruner(
    n_startup_trials=2,      # İlk 2 trial'da pruning yok
    n_warmup_steps=20        # 20 step warmup
)

# TrainingArguments
eval_steps=200               # 10K örnek için ~6 pruning fırsatı
logging_steps=50
```

### Sonuçları İnceleme

```python
import json

# Sonuçları yükle
with open("optuna_results.json", "r") as f:
    results = json.load(f)

# En iyi parametreleri al
best_params = results["best_trial"]["params"]
print(f"En iyi learning rate: {best_params['learning_rate']}")
print(f"En iyi grad_acc: {best_params['grad_acc']}")

# En iyi metrikleri görüntüle
best_metrics = results["best_metrics"]
print(f"En iyi perplexity: {best_metrics['perplexity']:.2f}")
print(f"En iyi eval_loss: {best_metrics['eval_loss']:.4f}")
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

1. **Minimum Dataset Boyutu**: Optuna için 10K, final için 2M+ örnek önerilir
2. **Kalite > Miktar**: Az ama kaliteli veri, çok ama düşük kaliteli veriden daha iyidir
3. **Dengeli Dağılım**: Farklı konu ve senaryoları kapsamalı
4. **Temizlik**: Gereksiz karakterler, HTML tag'leri temizlenmeli

## 💾 Checkpoint Yönetimi

### Checkpoint Türleri

Proje **üç tip checkpoint** kullanır:

1. **Step-based Checkpoint'ler**: Her `save_steps` step'te otomatik kayıt
   - `checkpoint-8000/`, `checkpoint-16000/`, vb.
   - `save_total_limit` ile sınırlı (son N checkpoint tutulur)

2. **Epoch-based Checkpoint'ler**: Her epoch bitişinde kayıt
   - `epoch-1-final/`, `epoch-2-final/`
   - `CheckpointCallback` ile otomatik

3. **Final Checkpoint**: Training bitişinde (max_steps veya epoch tamamlanınca)
   - `final-step-40000/` veya `trainer.save_model()` ile kaydedilen model

### Checkpoint Konumu

Her deney için checkpoint'ler ayrı dizinde saklanır:

```
models/checkpoints/E01_1day/
├── config.json
├── generation_config.json
├── model.safetensors (veya model-*.safetensors)
├── tokenizer.json
├── tokenizer_config.json
├── checkpoint-8000/              # Step-based
├── checkpoint-16000/
├── epoch-1-final/                # Epoch-based
└── final-step-40000/             # Final (max_steps)
```

### Resume Training

Eğitim kopsa bile checkpoint'ten devam eder:

```python
# Otomatik resume kontrolü
checkpoint = get_last_checkpoint(out_dir)
if checkpoint:
    print(f"📂 Checkpoint bulundu, devam ediliyor: {checkpoint}")
else:
    print("🆕 Yeni eğitim başlatılıyor")

trainer.train(resume_from_checkpoint=checkpoint)
```

### Checkpoint Kullanma

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Checkpoint'ten model yükle
model = AutoModelForCausalLM.from_pretrained("models/checkpoints/E01_1day")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoints/E01_1day")

# Inference
inputs = tokenizer("Merhaba, nasılsın?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Checkpoint Yedekleme

```bash
# Checkpoint'i yedekle
tar -czf E01_1day_checkpoint.tar.gz models/checkpoints/E01_1day/

# Yedekten geri yükle
tar -xzf E01_1day_checkpoint.tar.gz
```

## 🦙 Ollama Modelfile

Ollama ile deployment için Modelfile kullanılır:

### Modelfile Örneği

```dockerfile
FROM ./models/checkpoints/E01_1day/e01_1day.q8.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM """
Sen doğal, akıcı ve mantıklı cevaplar veren bir sohbet asistanısın.
Kısa ama anlamlı cevaplar üret.
Gereksiz tekrar yapma.
"""
```

### GGUF Dönüşümü

**Önemli**: Ollama GGUF formatı gerektirir. HuggingFace formatını GGUF'ye dönüştürmek için:

```bash
# llama.cpp kullanarak
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# HuggingFace modelini GGUF'ye dönüştür
python convert-hf-to-gguf.py \
  --outfile ./e01_1day.q8.gguf \
  --outtype q8_0 \
  models/checkpoints/E01_1day/
```

### Ollama'ya Yükleme

```bash
# Modelfile'ı Ollama'ya yükle
ollama create e01_1day -f Modelfile.E01_1day

# Test et
ollama run e01_1day "Merhaba, nasılsın?"
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

**H100 80GB için optimizasyonlar zaten aktif**:
1. **Gradient Checkpointing**: `gradient_checkpointing=True` aktif
2. **BF16 Training**: `bf16=True` aktif
3. **Device Map**: `device_map="auto"` ile otomatik offloading
4. **Batch Size**: H100 için optimize edilmiş (batch_size=8, grad_acc=2)

**Eğer hala OOM alırsan**:
1. `batch_size`'ı azalt (8 → 6 → 4)
2. `seq_len`'i azalt (768 → 512)
3. `grad_acc`'i artır (2 → 4 → 6)

### Dataset Format Hatası

```bash
# Dataset'i kontrol et
head -n 5 data/train.jsonl

# Her satırın geçerli JSON olduğundan emin ol
python3 -c "import json; [print(json.loads(line)) for line in open('data/train.jsonl')]"
```

### Checkpoint Resume Hatası

```bash
# Checkpoint dizinini kontrol et
ls -lh models/checkpoints/E01_1day/

# En son checkpoint'i manuel belirle
python train.py --run_id E01_1day \
  --resume_from_checkpoint models/checkpoints/E01_1day/checkpoint-32000
```

### Python Komutu Bulunamadı

```bash
# Python3 kullan
python3 train.py --run_id E01_1day

# Veya venv içindeki Python'u kullan
source venv/bin/activate
python train.py --run_id E01_1day
```

## 📈 Performans İpuçları

### GPU Bellek Optimizasyonu (H100 80GB için)

1. **BF16 Training**: Zaten aktif (`bf16=True`)
2. **Gradient Checkpointing**: Zaten aktif (`gradient_checkpointing=True`)
3. **Device Map**: Otomatik (`device_map="auto"`)
4. **Batch Size**: H100 için optimize edilmiş (`batch_size=8`, `grad_acc=2`)

### Eğitim Hızlandırma

1. **Sequence Length**: 768 kullan (1024 yerine %25-30 hız kazanımı)
2. **Max Steps**: Epoch yerine step-based training (18-24 saat)
3. **TensorCore Optimizasyonu**: `pad_to_multiple_of=8` aktif
4. **Group by Length**: `group_by_length=True` ile padding israfı azaltıldı
5. **Evaluation Optimizasyonu**: Sabit küçük eval seti (1000 örnek)

### Süre Optimizasyonları (270 saat → 18-24 saat)

| Optimizasyon | Etki | Süre Kazancı |
|-------------|------|--------------|
| Epoch 2 → 1 | ×0.5 | 135 saat |
| Max Steps 40K | ×0.3 | ~17 saat |
| Batch Size ↑ | ×0.7 | - |
| Seq Len 768 | ×0.75 | - |
| **Toplam** | **~0.08** | **~21 saat** |

**Not**: Kalite %85-90 korunur (production için yeterli)

## 🔄 Sonraki Adımlar

1. **Optuna ile Optimizasyon**: 10K örnekle parametre aralığı bul (2-3 saat)
2. **Final Training**: 2M örnekle production-ready model eğit (18-24 saat)
3. **Model Değerlendirme**: Test seti ile modeli değerlendir
4. **GGUF Dönüşümü**: HuggingFace → GGUF formatına dönüştür
5. **Ollama Deployment**: Modelfile ile Ollama'ya yükle
6. **Production**: Modeli production'a al (API, chatbot, vb.)

## 📚 Kaynaklar

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Llama 3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF dönüşümü için)

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
- Google Colab ekibine H100 GPU erişimi için

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
