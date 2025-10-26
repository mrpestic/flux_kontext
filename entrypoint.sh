#!/bin/bash
set -e

echo "⚙️ Подготавливаем окружение..."

# Настраиваем кэш HF на volume
export HF_HOME="/runpod-volume/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/runpod-volume/.cache/huggingface"
export TRANSFORMERS_CACHE="/runpod-volume/.cache/huggingface"
export DIFFUSERS_CACHE="/runpod-volume/.cache/huggingface"

mkdir -p "$HF_HOME"
mkdir -p "/runpod-volume/models"

# Логинимся в HF если есть токен
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Логинимся в Hugging Face Hub..."
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    huggingface-cli login --token "$HF_TOKEN" || true
fi

echo "🚀 Запускаем handler (пайплайн загрузится при импорте)..."
python3 /app/handler.py
