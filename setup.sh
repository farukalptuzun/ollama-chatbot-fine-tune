#!/bin/bash

# Sunucu kurulum scripti
# Kullanım: bash setup.sh

set -e  # Hata durumunda durdur

echo "🚀 Sunucu kurulumu başlatılıyor..."
echo ""

# Python versiyon kontrolü
echo "📋 Python versiyonu kontrol ediliyor..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı! Lütfen Python3 kurun."
    exit 1
fi
python3 --version
echo ""

# Virtual environment oluştur (yoksa)
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
    echo "✅ Virtual environment oluşturuldu"
else
    echo "✅ Virtual environment zaten mevcut"
fi
echo ""

# Virtual environment'ı aktifleştir
echo "🔄 Virtual environment aktifleştiriliyor..."
source venv/bin/activate
echo "✅ Virtual environment aktif"
echo ""

# pip'i güncelle
echo "⬆️  pip güncelleniyor..."
pip install --upgrade pip setuptools wheel --quiet
echo "✅ pip güncellendi"
echo ""

# Requirements.txt kontrolü
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  UYARI: requirements.txt bulunamadı!"
    echo "   Paketler kurulmayacak."
else
    # Gerekli paketleri kur
    echo "📚 Paketler kuruluyor (bu biraz zaman alabilir)..."
    echo "   Torch ve transformers gibi büyük paketler kuruluyor..."
    pip install -r requirements.txt
    echo "✅ Tüm paketler kuruldu"
fi
echo ""

# Gerekli dizinleri oluştur
echo "📁 Gerekli dizinler oluşturuluyor..."
mkdir -p models/checkpoints
mkdir -p data
mkdir -p logs
echo "✅ Dizinler oluşturuldu:"
echo "   - models/checkpoints/"
echo "   - data/"
echo "   - logs/"
echo ""

# runs.yaml kontrolü
if [ ! -f "runs.yaml" ]; then
    echo "⚠️  UYARI: runs.yaml dosyası bulunamadı!"
    echo "   Eğitim için runs.yaml dosyası gerekli."
else
    echo "✅ runs.yaml mevcut"
fi
echo ""

# data/train.jsonl kontrolü
if [ ! -f "data/train.jsonl" ]; then
    echo "⚠️  UYARI: data/train.jsonl dosyası bulunamadı!"
    echo "   Eğitim için dataset dosyası gerekli."
else
    echo "✅ data/train.jsonl mevcut"
fi
echo ""

# HuggingFace token kontrolü
echo "ℹ️  HATIRLATMA: HuggingFace model kullanıyorsanız token gerekebilir:"
echo "   huggingface-cli login"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "✅ Kurulum tamamlandı!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📝 Sonraki adımlar:"
echo "   1. source venv/bin/activate"
echo "   2. huggingface-cli login  (gerekirse)"
echo "   3. bash run_experiment.sh <RUN_ID>"
echo ""
