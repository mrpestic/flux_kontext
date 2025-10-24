FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка Python зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Установка FLUX из официального репозитория
RUN pip3 install git+https://github.com/black-forest-labs/flux.git

# Копирование исходного кода
COPY . .

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Открытие порта для RunPod
EXPOSE 8000

# Команда запуска
CMD ["python3", "handler.py"]
