# FLUX.1 Kontext RunPod Serverless API

Серверлесс API для редактирования изображений с помощью FLUX.1 Kontext на RunPod.

## Описание

Этот проект предоставляет серверлесс эндпоинт для редактирования изображений с использованием модели FLUX.1 Kontext от Black Forest Labs. Веса модели загружаются автоматически при первом запуске воркера.

## Структура проекта

```
flux_kontext/
├── Dockerfile          # Docker образ для RunPod
├── handler.py          # Основной хендлер с API эндпоинтами
├── requirements.txt    # Python зависимости
└── README.md          # Этот файл
```

## Настройка в RunPod

1. **Создание серверлесс эндпоинта:**
   - Перейдите в [RunPod Console](https://runpod.io/console/serverless)
   - Нажмите "New Endpoint"
   - Выберите "From GitHub"

2. **Конфигурация:**
   - **GitHub Repository:** `your-username/flux_kontext`
   - **Dockerfile Path:** `./Dockerfile`
   - **Container Port:** `8000`
   - **Container Disk:** `50 GB` (минимум для модели)
   - **Startup Disk:** `50 GB`
   - **GPU:** `RTX 4090` или выше (рекомендуется)

3. **Переменные окружения:**
   ```
   PYTHONPATH=/app
   CUDA_VISIBLE_DEVICES=0
   HF_TOKEN=your_huggingface_token_here
   ```

## API Эндпоинты

### POST `/generate`
Генерация множественных изображений из base64.

**Тело запроса (JSON):**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "prompt": "Add a hat to the person",
  "num_images": 3,
  "width": 1024,
  "height": 1024,
  "guidance_scale": 2.5,
  "num_inference_steps": 20
}
```

**Параметры:**
- `image_base64` (string) - изображение в формате base64
- `prompt` (string) - текстовое описание изменений
- `num_images` (int) - количество изображений для генерации (1-10)
- `width` (int) - ширина изображения (64-2048)
- `height` (int) - высота изображения (64-2048)
- `guidance_scale` (float) - сила следования промпту (по умолчанию: 2.5)
- `num_inference_steps` (int) - количество шагов инференса (по умолчанию: 20)

**Пример запроса:**
```bash
curl -X POST "https://your-endpoint.runpod.net/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "prompt": "Add a hat to the person",
    "num_images": 3,
    "width": 1024,
    "height": 1024
  }'
```

### GET `/health`
Проверка состояния сервиса.

**Ответ:**
```json
{
  "status": "healthy",
  "message": "Сервис работает"
}
```

## Ответы API

Эндпоинт `/generate` возвращает JSON с полями:

```json
{
  "success": true,
  "images_base64": [
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  ],
  "processing_time": 45.67,
  "seeds_used": [1234567890, 9876543210, 5555555555],
  "error": null
}
```

**Поля ответа:**
- `success` (bool) - успешность операции
- `images_base64` (array) - массив изображений в base64
- `processing_time` (float) - время обработки в секундах
- `seeds_used` (array) - массив использованных seed'ов для каждого изображения
- `error` (string|null) - описание ошибки если есть

## Требования к системе

- **GPU:** RTX 4090 или выше (24GB+ VRAM)
- **RAM:** 32GB+ рекомендуется
- **Диск:** 50GB+ для модели и зависимостей

## Особенности

- **Официальный FLUX API:** Используется официальный пакет `flux` из [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- **Автозагрузка модели:** Веса FLUX.1 Kontext загружаются автоматически при первом запуске с использованием HF_TOKEN
- **Множественная генерация:** Поддержка генерации до 10 изображений за один запрос
- **Случайные seed'ы:** Каждое изображение генерируется с уникальным случайным seed'ом
- **Настраиваемое разрешение:** Поддержка разрешений от 64x64 до 2048x2048
- **Оптимизация памяти:** Используется `torch.bfloat16` для экономии VRAM
- **Масштабируемость:** Поддержка до 10 одновременных инстансов

## Лицензия

Этот проект использует модель FLUX.1 Kontext под лицензией [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

## Поддержка

- [Документация FLUX.1 Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [RunPod Документация](https://docs.runpod.io/)
- [GitHub репозиторий FLUX](https://github.com/black-forest-labs/flux)
