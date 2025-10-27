#!/usr/bin/env python3
"""
Скрипт предзагрузки FLUX.1 Kontext пайплайна
Запускается при сборке образа или при первом старте
"""
import os
import logging
from diffusers import FluxKontextPipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_pipeline():
    """Предзагружает pipeline и сохраняет его в кэш"""
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("⚠️ HF_TOKEN не найден, пропускаем предзагрузку")
        return False
    
    cache_dir = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")
    model_name = "black-forest-labs/FLUX.1-Kontext-dev"
    
    logger.info(f"🚀 Начинаем предзагрузку модели {model_name}...")
    logger.info(f"📁 Кэш: {cache_dir}")
    
    try:
        # Загружаем pipeline (он автоматически кэшируется в cache_dir)
        pipeline = FluxKontextPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token,
            cache_dir=cache_dir,
            local_files_only=False,  # Скачиваем если нет в кэше
        )
        
        logger.info("✅ Пайплайн предзагружен и закэширован!")
        logger.info("📦 Модель будет доступна при запуске воркера")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при предзагрузке: {e}")
        return False

if __name__ == "__main__":
    preload_pipeline()

