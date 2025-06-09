# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями
COPY requirements_api.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

# Копируем исходный код приложения
COPY app.py .
COPY model.py .

# Копируем обученную модель и базу данных признаков
COPY best_flower_model.pth .
COPY flower_features_db.pkl .

# Создаем директорию для данных (опционально)
RUN mkdir -p /app/data

# Открываем порт для API
EXPOSE 8000

# Настройка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Опционально: переменная для использования GPU
ENV USE_GPU=false

# Команда запуска приложения
CMD ["python", "app.py"] 