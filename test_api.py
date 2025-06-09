#!/usr/bin/env python3
"""
Скрипт для тестирования Flower Similarity Search API
"""

import requests
import json
import os
import time
from pathlib import Path
import argparse


def test_health_check(base_url: str):
    """Тестирование health check эндпоинта"""
    print("Тестирование health check...")
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"Статус: {data['status']}")
        print(f"Изображений в базе: {data['images_count']}")
        print(f"Классы: {data['classes']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка health check: {e}")
        return False


def test_stats(base_url: str):
    """Тестирование статистики системы"""
    print("\nПолучение статистики системы...")
    
    try:
        response = requests.get(f"{base_url}/stats")
        response.raise_for_status()
        
        data = response.json()
        print(f"Всего изображений: {data['total_images']}")
        print(f"Размерность признаков: {data['feature_dimension']}")
        print(f"Устройство: {data['device']}")
        print("Распределение по классам:")
        for class_name, count in data['class_distribution'].items():
            print(f"   {class_name}: {count}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка получения статистики: {e}")
        return False


def test_search(base_url: str, image_path: str, top_k: int = 5):
    """Тестирование поиска похожих изображений"""
    print(f"\nТестирование поиска для {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            params = {'top_k': top_k, 'include_predictions': True}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/search", files=files, params=params)
            search_time = time.time() - start_time
            
            response.raise_for_status()
            
            data = response.json()
            print(f"Поиск завершен за {search_time:.2f} сек")
            print(f"Найдено {len(data['results'])} похожих изображений:")
            
            # Выводим результаты поиска
            for result in data['results']:
                print(f"   {result['rank']}. {result['image_path']}")
                print(f"      Сходство: {result['similarity_score']:.6f}")
                print(f"      Класс: {result['predicted_class']}")
            
            # Выводим предсказания классов
            if 'class_predictions' in data:
                print("\nПредсказания классов:")
                for class_name, prob in list(data['class_predictions'].items())[:3]:
                    print(f"   {class_name}: {prob:.4f} ({prob*100:.1f}%)")
            
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка поиска: {e}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False


def test_predict(base_url: str, image_path: str):
    """Тестирование предсказания класса"""
    print(f"\nТестирование предсказания класса для {image_path}...")
    
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(f"{base_url}/predict", files=files)
            response.raise_for_status()
            
            data = response.json()
            print(f"Предсказание завершено")
            print(f"Топ предсказание: {data['top_prediction']['class_name']} "
                  f"({data['top_prediction']['percentage']})")
            
            print("\nВсе предсказания:")
            for pred in data['predictions']:
                print(f"   {pred['class_name']}: {pred['percentage']}")
            
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"Ошибка предсказания: {e}")
        return False


def find_test_images(dataset_path: str, max_images: int = 3):
    """Поиск тестовых изображений в датасете"""
    test_images = []
    
    if not os.path.exists(dataset_path):
        print(f"Датасет не найден: {dataset_path}")
        return test_images
    
    # Ищем изображения в подпапках
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= max_images:
                    return test_images
    
    return test_images


def main():
    parser = argparse.ArgumentParser(description="Тестирование Flower Similarity Search API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Базовый URL API (по умолчанию: http://localhost:8000)")
    parser.add_argument("--image", help="Путь к изображению для тестирования")
    parser.add_argument("--dataset", default="flowers-recognition", 
                       help="Путь к датасету для поиска тестовых изображений")
    parser.add_argument("--top-k", type=int, default=5, 
                       help="Количество похожих изображений для поиска")
    
    args = parser.parse_args()
    
    print("Тестирование Flower Similarity Search API")
    print("=" * 60)
    
    # Проверка доступности API
    if not test_health_check(args.url):
        print("API недоступен. Убедитесь, что сервис запущен.")
        return
    
    # Получение статистики
    test_stats(args.url)
    
    # Определение тестовых изображений
    test_images = []
    
    if args.image:
        test_images = [args.image]
    else:
        # Ищем тестовые изображения в датасете
        test_images = find_test_images(args.dataset, max_images=2)
    
    if not test_images:
        print(f"\nТестовые изображения не найдены.")
        print(f"Укажите путь к изображению с помощью --image или проверьте папку {args.dataset}")
        return
    
    # Тестирование поиска и предсказания
    success_count = 0
    total_tests = len(test_images) * 2  # поиск + предсказание для каждого изображения
    
    for image_path in test_images:
        print(f"\n{'='*60}")
        print(f"Тестирование изображения: {Path(image_path).name}")
        print("=" * 60)
        
        # Тест поиска
        if test_search(args.url, image_path, args.top_k):
            success_count += 1
        
        # Тест предсказания
        if test_predict(args.url, image_path):
            success_count += 1
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print(f"ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Успешных тестов: {success_count}/{total_tests}")
    print(f"Успешность: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("Все тесты прошли успешно!")
    else:
        print("Некоторые тесты завершились с ошибками.")


if __name__ == "__main__":
    main() 