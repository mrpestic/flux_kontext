import runpod
import torch
import base64
import io
import logging
import random
import os
from PIL import Image
from diffusers import FluxKontextPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный пайплайн - загружаем при импорте модуля
pipeline = None

def get_pipeline():
    """Получаем предзагруженный пайплайн"""
    global pipeline
    if pipeline is None:
        logger.error("❌ Пайплайн не загружен!")
        raise RuntimeError("Pipeline not initialized")
    return pipeline

def load_image_from_base64(image_base64: str) -> Image.Image:
    """Загрузка изображения из base64"""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def image_to_base64(image: Image.Image) -> str:
    """Конвертация изображения в base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def handler(event):
    """Основной handler для RunPod"""
    import time
    start_time = time.time()
    
    try:
        input_data = event.get("input", {})
        image_base64 = input_data.get("image_base64")
        prompt = input_data.get("prompt")
        num_images = input_data.get("num_images", 1)
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        guidance_scale = input_data.get("guidance_scale", 2.5)
        num_inference_steps = input_data.get("num_inference_steps", 20)
        
        # Получаем предзагруженный пайплайн
        pipe = get_pipeline()
        
        # Загрузка и обработка изображения
        input_image = load_image_from_base64(image_base64)
        if input_image.size != (width, height):
            input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
        
        generated_images = []
        for i in range(num_images):
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            result = pipe(
                image=input_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            generated_images.append(result.images[0])
        
        images_base64 = [image_to_base64(img) for img in generated_images]
        
        return {
            "output": {
                "success": True,
                "images_base64": images_base64,
                "processing_time": time.time() - start_time
            }
        }
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return {"output": {"error": str(e)}}


# ЗАГРУЗКА ПАЙПЛАЙНА ПРИ ИМПОРТЕ МОДУЛЯ
# Это выполнится ОДИН РАЗ при старте воркера
logger.info("🚀 Загрузка FLUX.1 Kontext пайплайна...")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN не найден в переменных окружения")

logger.info("📥 Загружаем модель в GPU...")
pipeline = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token=hf_token,
    low_cpu_mem_usage=True
).to("cuda")

logger.info("✅ Пайплайн загружен в GPU память!")
logger.info("🎯 Handler готов к работе, запускаем RunPod serverless...")

# Запускаем RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


