# preload.py
#!/usr/bin/env python3
import os
import logging
from diffusers import FluxKontextPipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    cache_dir = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")

    logger.info(f"🚀 Начинаем предзагрузку весов {model_id}")
    logger.info(f"📁 Кэш: {cache_dir}")

    try:
        # Скачиваем и кладём в кэш. На GPU НЕ переносим — VRAM не трогаем.
        _ = FluxKontextPipeline.from_pretrained(
            model_id,
            dtype=torch.float16,        # новый параметр вместо устаревшего torch_dtype
            cache_dir=cache_dir,
            local_files_only=False,     # если не скачано — скачается сейчас
            low_cpu_mem_usage=True
        )
        logger.info("✅ Веса предзагружены и закэшированы.")
    except Exception as e:
        logger.error(f"❌ Ошибка предзагрузки: {e}")
        raise

if __name__ == "__main__":
    main()