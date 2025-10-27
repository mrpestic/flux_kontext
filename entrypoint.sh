#!/bin/bash
set -euo pipefail

echo "⚙️  Подготавливаем окружение..."

# Кэш Hugging Face на персистентном volume
export HF_HOME="/runpod-volume/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export DIFFUSERS_CACHE="$HF_HOME"

mkdir -p "$HF_HOME"
mkdir -p "/runpod-volume/models"

# Логин в HF (если токен задан в переменных окружения панели RunPod)
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "🔑 Логинимся в Hugging Face Hub..."
  # CLI может быть разным в зависимости от версии пакета; игнорируем предупреждения
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

# Предзагрузка весов (скачивание в кэш, без загрузки в VRAM)
echo "🚀 Предзагружаем веса модели (кэш HF)..."
python3 -u /app/preload.py || true

# ВАЖНО: запускаем один-единственный процесс Python, который откроет serverless-луп
echo "🎯 Запускаем RunPod serverless handler..."
exec python3 -u /app/handler.py