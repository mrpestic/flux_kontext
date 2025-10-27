# handler.py
import os
import io
import time
import base64
import logging
import random
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import torch
from PIL import Image
from diffusers import FluxKontextPipeline
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- УТИЛИТЫ ----------

def _b64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img

def _load_image_from_path_or_url(path_or_url: str) -> Image.Image:
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        req = Request(path_or_url, headers={"User-Agent": "RunPod/Serverless"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    # локальный путь
    with open(path_or_url, "rb") as f:
        data = f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def _pick_input_image(inp: dict) -> Image.Image:
    """
    Поддерживает и image_base64, и image_path (URL/локальный путь).
    """
    if "image_base64" in inp and inp["image_base64"]:
        return _b64_to_image(inp["image_base64"])
    if "image_path" in inp and inp["image_path"]:
        return _load_image_from_path_or_url(inp["image_path"])
    raise ValueError("Нужно передать image_base64 или image_path")

def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- ГЛОБАЛЬНАЯ МОДЕЛЬ (ГРУЗИМ ПРИ СТАРТЕ ВОРКЕРА) ----------

PIPELINE: Optional[FluxKontextPipeline] = None

def _load_pipeline_once() -> FluxKontextPipeline:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    cache_dir = os.getenv("HF_HOME", "/runpod-volume/.cache/huggingface")

    logger.info("🚀 Загрузка FLUX.1-Kontext пайплайна из кэша (инициализация воркера)...")
    try:
        # Важно: не передаём use_auth_token (он игнорируется в этой реализации),
        # аутентификация идёт через окружение/логин.
        PIPELINE = FluxKontextPipeline.from_pretrained(
            model_id,
            dtype=torch.float16,         # вместо torch_dtype
            cache_dir=cache_dir,
            local_files_only=True,       # ТОЛЬКО из кэша; качали в preload.py
            low_cpu_mem_usage=True
        ).to("cuda")
        torch.set_grad_enabled(False)
        logger.info("✅ Пайплайн загружен в GPU.")
        return PIPELINE
    except Exception as e:
        # Если кэш почему-то пуст — лучше упасть явной ошибкой, чем качать в задаче (чтобы не было биллинга на загрузку)
        logger.exception("❌ Не удалось загрузить пайплайн из кэша. Убедись, что preload прошёл успешно.")
        raise

# Инициализация при старте воркера (cold start), до получения задач
PIPELINE = _load_pipeline_once()

# ---------- RUNPOD HANDLER ----------

def handler(event):
    t0 = time.time()
    try:
        inp = event.get("input")
        if not isinstance(inp, dict):
            return {"error": "payload должен содержать поле 'input' с параметрами"}

        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "нужно поле 'prompt' (строка)"}

        # Параметры с дефолтами
        num_images = int(inp.get("num_images", 1))
        steps = int(inp.get("num_inference_steps", 28))  # FLUX оптимизирован для 28 шагов
        guidance = float(inp.get("guidance_scale", inp.get("guidance", 3.5)))  # FLUX использует 3.5
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))

        # Входная картинка (из base64 или пути/URL)
        init_img = _pick_input_image(inp)

        pipe = _load_pipeline_once()

        images = []
        for _ in range(num_images):
            torch.manual_seed(random.randint(0, 2**32 - 1))
            out = pipe(
                image=init_img,
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width
            ).images[0]
            images.append(_image_to_b64(out))

        return {
            "output": {
                "success": True,
                "images_base64": images,
                "elapsed": time.time() - t0
            }
        }
    except Exception as e:
        logger.exception("❌ Ошибка выполнения handler")
        return {"output": {"error": str(e)}}

# Запуск serverless-воркера (важно вызывать без условий)
runpod.serverless.start({"handler": handler})