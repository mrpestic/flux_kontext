FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка Python зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Установка переменных окружения
ENV PYTHONPATH=/app:/app/flux
ENV CUDA_VISIBLE_DEVICES=0

# Проброс порта
EXPOSE 8000

# Запуск Flask сервера
CMD ["python3", "handler.py"]
