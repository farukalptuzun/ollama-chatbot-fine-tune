#!/bin/bash
RUN_ID=$1

echo "🔍 Script başlatılıyor, RUN_ID: $RUN_ID" >&2

# Venv aktif değilse aktifleştir
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "📦 Venv aktifleştiriliyor..." >&2
        source venv/bin/activate
    else
        echo "⚠️  Venv dizini bulunamadı!" >&2
    fi
else
    echo "✅ Venv zaten aktif" >&2
fi

echo "🐍 Python çalıştırılıyor..." >&2
python3 train.py \
  --run_id $RUN_ID \
  --model_name meta-llama/Meta-Llama-3-8B \
  --dataset data/train.jsonl

