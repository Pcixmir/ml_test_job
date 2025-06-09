from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn

from model import FlowerSearchSystem

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Flower Similarity Search API",
    description="API для поиска похожих изображений цветов",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для системы поиска
search_system: Optional[FlowerSearchSystem] = None

def initialize_search_system():
    """Инициализация системы поиска при запуске приложения"""
    global search_system
    
    try:
        model_path = "best_flower_model.pth"
        features_db_path = "flower_features_db.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        if not os.path.exists(features_db_path):
            raise FileNotFoundError(f"База данных признаков не найдена: {features_db_path}")
        
        # Определяем устройство (CPU/GPU)
        device = 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu'
        
        search_system = FlowerSearchSystem(
            model_path=model_path,
            features_db_path=features_db_path,
            device=device
        )
        
        logger.info("✅ Система поиска успешно инициализирована")
        logger.info(f"🖥️  Устройство: {device}")
        logger.info(f"🗃️  Изображений в базе: {len(search_system.image_database)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации системы поиска: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("🚀 Запуск Flower Similarity Search API...")
    initialize_search_system()

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API"""
    return {
        "message": "Flower Similarity Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "POST - Поиск похожих изображений цветов",
            "/predict": "POST - Предсказание класса цветка",
            "/health": "GET - Проверка состояния сервиса",
            "/stats": "GET - Статистика системы"
        },
        "usage": {
            "search": "Отправьте POST запрос с изображением на /search",
            "formats": "Поддерживаемые форматы: JPEG, PNG, GIF, BMP",
            "response": "JSON с путями к похожим изображениям и их показателями сходства"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Система поиска не инициализирована")
    
    return {
        "status": "healthy",
        "model_loaded": search_system.model is not None,
        "database_loaded": search_system.features_database is not None,
        "images_count": len(search_system.image_database) if search_system.image_database else 0,
        "classes": list(search_system.label_to_idx.keys()) if search_system.label_to_idx else []
    }

@app.get("/stats")
async def get_stats():
    """Получение статистики системы"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Система поиска не инициализирована")
    
    # Подсчет изображений по классам
    class_counts = {}
    for image_path in search_system.image_database:
        # Извлекаем класс из пути (предполагаем структуру .../class_name/image.jpg)
        path_parts = Path(image_path).parts
        if len(path_parts) >= 2:
            class_name = path_parts[-2]  # Предпоследняя часть пути
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "total_images": len(search_system.image_database),
        "classes": list(search_system.label_to_idx.keys()),
        "class_distribution": class_counts,
        "feature_dimension": search_system.model.feature_dim if search_system.model else None,
        "device": str(search_system.device)
    }

def validate_image(file: UploadFile) -> Image.Image:
    """Валидация и загрузка изображения"""
    # Проверка типа файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый тип файла: {file.content_type}. Поддерживаются только изображения."
        )
    
    # Проверка размера файла (максимум 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Файл слишком большой: {file.size} байт. Максимальный размер: {max_size} байт."
        )
    
    try:
        # Загрузка и конвертация изображения
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Конвертация в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )

@app.post("/search")
async def search_similar_flowers(
    file: UploadFile = File(...),
    top_k: int = 5,
    include_predictions: bool = False
):
    """
    Поиск похожих изображений цветов
    
    Args:
        file: Загружаемое изображение
        top_k: Количество похожих изображений для возврата (1-20)
        include_predictions: Включить предсказания классов в ответ
    
    Returns:
        JSON с результатами поиска
    """
    if search_system is None:
        raise HTTPException(status_code=503, detail="Система поиска не инициализирована")
    
    # Валидация параметров
    if not (1 <= top_k <= 20):
        raise HTTPException(
            status_code=400,
            detail="top_k должно быть от 1 до 20"
        )
    
    try:
        # Валидация и загрузка изображения
        image = validate_image(file)
        
        logger.info(f"🔍 Поиск похожих изображений для {file.filename}, top_k={top_k}")
        
        # Поиск похожих изображений
        similar_images = search_system.search_similar_flowers(image, top_k=top_k)
        
        # Подготовка результата
        result = {
            "query_image": file.filename,
            "top_k": top_k,
            "results": []
        }
        
        # Добавление результатов поиска
        for rank, (image_path, similarity_score) in enumerate(similar_images.items(), 1):
            # Извлекаем относительный путь и класс
            try:
                relative_path = str(Path(image_path).relative_to(Path.cwd()))
            except ValueError:
                # Если путь уже относительный или в другой папке
                relative_path = str(Path(image_path))
            
            class_name = Path(image_path).parts[-2] if len(Path(image_path).parts) >= 2 else "unknown"
            
            result_item = {
                "rank": rank,
                "image_path": relative_path,
                "similarity_score": round(similarity_score, 6),
                "predicted_class": class_name
            }
            
            result["results"].append(result_item)
        
        # Добавление предсказаний классов если запрошено
        if include_predictions:
            class_predictions = search_system.predict_class(image)
            result["class_predictions"] = {
                class_name: round(prob, 6) 
                for class_name, prob in sorted(class_predictions.items(), 
                                             key=lambda x: x[1], reverse=True)
            }
        
        # Добавление метаданных
        result["metadata"] = {
            "total_database_images": len(search_system.image_database),
            "search_time_info": "Время поиска зависит от размера базы данных",
            "similarity_metric": "cosine_similarity",
            "model_architecture": "ResNet50 + Custom Feature Extractor"
        }
        
        logger.info(f"✅ Поиск завершен для {file.filename}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка поиска: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@app.post("/predict")
async def predict_flower_class(file: UploadFile = File(...)):
    """
    Предсказание класса цветка
    
    Args:
        file: Загружаемое изображение
    
    Returns:
        JSON с предсказаниями классов
    """
    if search_system is None:
        raise HTTPException(status_code=503, detail="Система поиска не инициализирована")
    
    try:
        # Валидация и загрузка изображения
        image = validate_image(file)
        
        logger.info(f"🔮 Предсказание класса для {file.filename}")
        
        # Получение предсказаний
        class_predictions = search_system.predict_class(image)
        
        # Сортировка по убыванию вероятности
        sorted_predictions = sorted(
            class_predictions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        result = {
            "query_image": file.filename,
            "predictions": [
                {
                    "class_name": class_name,
                    "probability": round(prob, 6),
                    "percentage": f"{round(prob * 100, 2)}%"
                }
                for class_name, prob in sorted_predictions
            ],
            "top_prediction": {
                "class_name": sorted_predictions[0][0],
                "probability": round(sorted_predictions[0][1], 6),
                "percentage": f"{round(sorted_predictions[0][1] * 100, 2)}%"
            }
        }
        
        logger.info(f"✅ Предсказание завершено для {file.filename}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    ) 