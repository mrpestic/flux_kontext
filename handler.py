import torch
import base64
import io
import logging
import random
import os
from typing import Optional, Dict, Any, List
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import numpy as np

# Импорт FLUX из diffusers
from diffusers import FluxKontextPipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для пайплайна
pipeline = None

class ImageEditRequest(BaseModel):
    image_base64: str
    prompt: str
    num_images: int = 1
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 2.5
    num_inference_steps: int = 20

class ImageEditResponse(BaseModel):
    success: bool
    images_base64: List[str] = []
    error: Optional[str] = None
    processing_time: Optional[float] = None
    seeds_used: List[int] = []

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_pipeline()
    yield
    # Shutdown
    pass

app = FastAPI(title="FLUX.1 Kontext API", version="1.0.0", lifespan=lifespan)

async def load_pipeline():
    """Загрузка пайплайна FLUX.1 Kontext"""
    global pipeline
    try:
        logger.info("Загрузка FLUX.1 Kontext пайплайна...")
        
        # Получаем токен из переменных окружения
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN не найден в переменных окружения")
        
        # Используем официальный API FLUX
        pipeline = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            dtype=torch.bfloat16,
            device_map="cuda",
            use_auth_token=hf_token
        )
        
        # Перемещаем на GPU
        pipeline.to("cuda")
        logger.info("Пайплайн успешно загружен!")
    except Exception as e:
        logger.error(f"Ошибка загрузки пайплайна: {e}")
        raise e

def load_image_from_base64(image_base64: str) -> Image.Image:
    """Загрузка изображения из base64"""
    try:
        # Убираем префикс data:image/...;base64, если есть
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Конвертация в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Ошибка загрузки изображения из base64: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки изображения: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """Конвертация изображения в base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Startup event заменен на lifespan в определении app

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {"message": "FLUX.1 Kontext API готов к работе", "status": "running"}

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Пайплайн не загружен"}
        )
    return {"status": "healthy", "message": "Сервис работает"}

@app.post("/generate", response_model=ImageEditResponse)
async def generate_images(request: ImageEditRequest):
    """Генерация множественных изображений из base64"""
    import time
    start_time = time.time()
    
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Пайплайн не загружен")
        
        # Валидация параметров
        if request.num_images < 1 or request.num_images > 10:
            raise HTTPException(status_code=400, detail="Количество изображений должно быть от 1 до 10")
        
        if request.width < 64 or request.height < 64 or request.width > 2048 or request.height > 2048:
            raise HTTPException(status_code=400, detail="Разрешение должно быть от 64x64 до 2048x2048")
        
        # Загрузка изображения
        input_image = load_image_from_base64(request.image_base64)
        
        # Изменение размера изображения если нужно
        if input_image.size != (request.width, request.height):
            input_image = input_image.resize((request.width, request.height), Image.Resampling.LANCZOS)
        
        generated_images = []
        seeds_used = []
        
        logger.info(f"Начинаем генерацию {request.num_images} изображений с промптом: {request.prompt}")
        
        # Генерация множественных изображений
        for i in range(request.num_images):
            # Генерация случайного seed для каждого изображения
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            seeds_used.append(seed)
            
            logger.info(f"Генерируем изображение {i+1}/{request.num_images} с seed {seed}")
            
            result = pipeline(
                image=input_image,
                prompt=request.prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps
            )
            
            generated_image = result.images[0]
            generated_images.append(generated_image)
        
        # Конвертация всех изображений в base64
        images_base64 = [image_to_base64(img) for img in generated_images]
        
        processing_time = time.time() - start_time
        logger.info(f"Генерация {request.num_images} изображений завершена за {processing_time:.2f} секунд")
        
        return ImageEditResponse(
            success=True,
            images_base64=images_base64,
            processing_time=processing_time,
            seeds_used=seeds_used
        )
        
    except Exception as e:
        logger.error(f"Ошибка при генерации изображений: {e}")
        return ImageEditResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )

if __name__ == "__main__":
    uvicorn.run(
        "handler:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
