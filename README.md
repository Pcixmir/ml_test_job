# 🌸 Flower Similarity Search API

Система поиска похожих изображений цветов на основе глубокого обучения с REST API и Docker-контейнеризацией.

## 📋 Описание

Этот проект предоставляет REST API для поиска похожих изображений цветов. Система обучена на датасете [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data) и может:

- 🔍 **Искать похожие изображения** по загруженному фото
- 🎯 **Предсказывать класс цветка** (daisy, dandelion, rose, sunflower, tulip)
- 📊 **Возвращать результаты в JSON** с показателями сходства
- 🐳 **Работать в Docker** контейнере
- ⚡ **Поддерживать GPU** ускорение (опционально)

## 🏗️ Архитектура

- **Модель**: ResNet50 + Custom Feature Extractor (512-dim features)
- **API**: FastAPI с автоматической документацией
- **Метрика сходства**: Косинусное сходство
- **Контейнеризация**: Docker + Docker Compose
- **Веб-сервер**: Nginx для статических файлов и проксирования

## 📁 Структура проекта

```
├── app.py                      # FastAPI приложение
├── model.py                    # Определение модели и системы поиска
├── best_flower_model.pth       # Обученная модель (282MB)
├── flower_features_db.pkl      # База данных признаков (8.7MB)
├── flowers-recognition/        # Датасет изображений
├── Dockerfile                  # Docker образ
├── docker-compose.yml          # Оркестрация контейнеров
├── nginx.conf                  # Конфигурация Nginx
├── requirements_api.txt        # Зависимости для API
├── test_api.py                 # Скрипт тестирования API
└── README.md                   # Этот файл
```

## 🚀 Быстрый старт

### 1. Проверка зависимостей

Убедитесь, что у вас установлены:
- Docker (версия 20.10+)
- Docker Compose (версия 1.29+)
- Python 3.9+ (для тестирования)

### 2. Клонирование и запуск

```bash
# Убедитесь, что все файлы на месте
ls -la best_flower_model.pth flower_features_db.pkl

# Запуск с Docker Compose
docker-compose up --build -d

# Проверка статуса контейнеров
docker-compose ps
```

### 3. Проверка работы API

```bash
# Установка зависимостей для тестирования
pip install requests

# Запуск тестов
python test_api.py

# Или тестирование конкретного изображения
python test_api.py --image flowers-recognition/flowers/rose/rose_001.jpg
```

## 🔌 API Эндпоинты

### 1. **Поиск похожих изображений**

```bash
curl -X POST "http://localhost:8000/search" \
     -H "accept: application/json" \
     -F "file=@flower.jpg" \
     -F "top_k=5" \
     -F "include_predictions=true"
```

**Ответ:**
```json
{
  "query_image": "flower.jpg",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "image_path": "flowers-recognition/flowers/rose/rose_123.jpg",
      "similarity_score": 0.945612,
      "predicted_class": "rose"
    }
  ],
  "class_predictions": {
    "rose": 0.892,
    "tulip": 0.056,
    "daisy": 0.032
  },
  "metadata": {
    "total_database_images": 4317,
    "similarity_metric": "cosine_similarity",
    "model_architecture": "ResNet50 + Custom Feature Extractor"
  }
}
```

### 2. **Предсказание класса**

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@flower.jpg"
```

### 3. **Проверка состояния**

```bash
curl http://localhost:8000/health
```

### 4. **Статистика системы**

```bash
curl http://localhost:8000/stats
```

### 5. **Автоматическая документация**

Откройте в браузере: http://localhost:8000/docs

## 🐳 Docker команды

### Основные команды

```bash
# Сборка и запуск
docker-compose up --build -d

# Просмотр логов
docker-compose logs -f flower-search-api

# Остановка
docker-compose down

# Перезапуск
docker-compose restart
```

### Запуск только API (без Nginx)

```bash
# Сборка образа
docker build -t flower-search-api .

# Запуск контейнера
docker run -p 8000:8000 \
           -v $(pwd)/flowers-recognition:/app/flowers-recognition:ro \
           flower-search-api
```

### Запуск с GPU поддержкой

```bash
# Изменить в docker-compose.yml:
environment:
  - USE_GPU=true

# Убедиться что установлен nvidia-docker
docker run --gpus all -p 8000:8000 flower-search-api
```

## 🧪 Тестирование

### Автоматическое тестирование

```bash
# Полное тестирование
python test_api.py

# Тестирование конкретного изображения
python test_api.py --image path/to/flower.jpg --top-k 10

# Тестирование с другим URL
python test_api.py --url http://localhost:8000
```

### Ручное тестирование с curl

```bash
# Health check
curl http://localhost:8000/health

# Поиск похожих изображений
curl -X POST http://localhost:8000/search \
     -F "file=@test_flower.jpg" \
     -F "top_k=3"

# Предсказание класса
curl -X POST http://localhost:8000/predict \
     -F "file=@test_flower.jpg"
```

## 🔧 Конфигурация

### Переменные окружения

```bash
# В docker-compose.yml или .env файле
USE_GPU=false                    # Использование GPU
PYTHONUNBUFFERED=1              # Вывод логов в реальном времени
```

### Настройка Nginx

Файл `nginx.conf` содержит:
- Проксирование API запросов
- Раздачу статических изображений
- Настройки кэширования
- Увеличенные таймауты для загрузки файлов

### Настройка API

В `app.py` можно изменить:
- Максимальный размер загружаемых файлов (10MB)
- Таймауты обработки
- Уровень логирования

## 📊 Производительность

### Характеристики модели

- **Размер модели**: 282MB
- **База данных признаков**: 8.7MB (4317 изображений)
- **Размерность признаков**: 512
- **Время поиска**: ~100-500ms на CPU
- **Поддерживаемые форматы**: JPEG, PNG, GIF, BMP, TIFF

### Оптимизация

Для улучшения производительности:

1. **GPU ускорение**: `USE_GPU=true`
2. **Увеличение batch_size** для обработки признаков
3. **Кэширование результатов** на уровне приложения
4. **Использование ONNX** для inference

## 🛠️ Разработка

### Локальная разработка

```bash
# Установка зависимостей
pip install -r requirements_api.txt

# Запуск в режиме разработки
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Добавление новых эндпоинтов

1. Добавьте функцию в `app.py`
2. Используйте декораторы FastAPI (`@app.get`, `@app.post`)
3. Добавьте документацию и валидацию
4. Обновите тесты в `test_api.py`

### Обновление модели

1. Обучите новую модель и сохраните как `best_flower_model.pth`
2. Пересоздайте базу данных признаков `flower_features_db.pkl`
3. Пересоберите Docker образ: `docker-compose build`

## 🔍 Устранение неполадок

### API не запускается

```bash
# Проверка логов
docker-compose logs flower-search-api

# Проверка доступности файлов модели
ls -la best_flower_model.pth flower_features_db.pkl

# Проверка портов
netstat -tlpn | grep 8000
```

### Ошибки загрузки изображений

- Убедитесь, что размер файла < 10MB
- Проверьте формат изображения (JPEG, PNG, etc.)
- Проверьте права доступа к файлу

### Низкая производительность

- Включите GPU: `USE_GPU=true`
- Увеличьте ресурсы Docker
- Проверьте нагрузку на систему: `htop`

## 📝 API Примеры

### Python

```python
import requests

# Поиск похожих изображений
with open('flower.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/search',
        files={'file': f},
        params={'top_k': 5, 'include_predictions': True}
    )
    
result = response.json()
print(f"Найдено {len(result['results'])} похожих изображений")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('top_k', '5');

fetch('http://localhost:8000/search', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
# Простой поиск
curl -X POST http://localhost:8000/search \
     -F "file=@flower.jpg" \
     -F "top_k=5"

# С предсказаниями
curl -X POST http://localhost:8000/search \
     -F "file=@flower.jpg" \
     -F "top_k=5" \
     -F "include_predictions=true"
```

## 🤝 Развитие проекта

### Планы развития

- [ ] Поддержка видео файлов
- [ ] Batch API для обработки множественных изображений
- [ ] Интеграция с S3/MinIO для хранения изображений
- [ ] Метрики и мониторинг (Prometheus + Grafana)
- [ ] A/B тестирование разных моделей
- [ ] Поддержка пользовательских датасетов

### Вклад в проект

1. Fork репозитория
2. Создайте feature branch: `git checkout -b feature/new-feature`
3. Сделайте изменения и добавьте тесты
4. Создайте Pull Request

## 📄 Лицензия

Этот проект создан в учебных целях. Модель обучена на открытом датасете Flowers Recognition.

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте [раздел устранения неполадок](#-устранение-неполадок)
2. Просмотрите логи: `docker-compose logs`
3. Запустите тесты: `python test_api.py`
4. Проверьте документацию API: http://localhost:8000/docs 