import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional


class FlowerFeatureExtractor(nn.Module):
    """Модель для извлечения признаков изображений цветов"""
    
    def __init__(self, backbone='resnet50', num_classes=5, feature_dim=512):
        super(FlowerFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Загружаем предобученную модель
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Убираем последний слой
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            backbone_output_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=True)
            backbone_output_dim = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
            
        else:
            raise ValueError(f"Неподдерживаемая архитектура: {backbone}")
        
        # Создаем слои для извлечения признаков и классификации
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Классификационная головка
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов новых слоев"""
        for m in [self.feature_extractor, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_features=False):
        """Прямой проход через модель"""
        # Извлекаем признаки из backbone
        backbone_features = self.backbone(x)
        
        # Получаем финальные признаки
        features = self.feature_extractor(backbone_features)
        
        if return_features:
            return features
        
        # Классификация
        logits = self.classifier(features)
        
        return logits, features
    
    def extract_features(self, x):
        """Извлечение нормализованных признаков для поиска похожих изображений"""
        with torch.no_grad():
            features = self.forward(x, return_features=True)
            # L2 нормализация для косинусного сходства
            features = nn.functional.normalize(features, p=2, dim=1)
        return features


class FlowerSearchSystem:
    """Система поиска похожих изображений цветов"""
    
    def __init__(self, model_path: str, features_db_path: str, device: str = 'cpu'):
        """
        Инициализация системы поиска
        
        Args:
            model_path: Путь к сохраненной модели
            features_db_path: Путь к базе данных признаков
            device: Устройство для вычислений
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.image_database = []
        self.features_database = None
        
        # Загружаем модель и базу данных
        self.load_model(model_path)
        self.load_features_database(features_db_path)
        
        # Трансформация для новых изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        """Загрузка обученной модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Создаем модель с правильной архитектурой
        self.model = FlowerFeatureExtractor(
            backbone='resnet50',
            num_classes=len(self.label_to_idx),
            feature_dim=512
        ).to(self.device)
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Модель загружена с точностью {checkpoint['val_acc']:.2f}%")
        print(f"Классы: {list(self.label_to_idx.keys())}")
    
    def load_features_database(self, features_db_path: str):
        """Загрузка базы данных признаков"""
        if not os.path.exists(features_db_path):
            raise FileNotFoundError(f"База данных признаков не найдена: {features_db_path}")
        
        with open(features_db_path, 'rb') as f:
            data = pickle.load(f)
            self.image_database = data['image_paths']
            self.features_database = data['features']
        
        print(f"База данных признаков загружена: {len(self.image_database)} изображений")
    
    def search_similar_flowers(self, query_image: Image.Image, top_k: int = 5) -> Dict[str, float]:
        """
        Поиск похожих изображений цветов
        
        Args:
            query_image: PIL изображение запроса
            top_k: Количество похожих изображений для возврата
            
        Returns:
            dict: Словарь {image_path: similarity_score} отсортированный по убыванию сходства
        """
        if self.features_database is None:
            raise ValueError("База данных признаков не загружена!")
        
        # Обрабатываем изображение запроса
        try:
            query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Ошибка обработки изображения запроса: {e}")
        
        # Извлекаем признаки запроса
        query_features = self.model.extract_features(query_tensor).cpu().numpy()
        
        # Вычисляем косинусное сходство
        similarities = cosine_similarity(query_features, self.features_database)[0]
        
        # Получаем топ-k наиболее похожих
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Формируем результат
        results = {}
        for idx in top_indices:
            image_path = self.image_database[idx]
            similarity_score = float(similarities[idx])
            results[image_path] = similarity_score
        
        return results
    
    def predict_class(self, query_image: Image.Image) -> Dict[str, float]:
        """
        Предсказание класса изображения
        
        Args:
            query_image: PIL изображение
            
        Returns:
            dict: Словарь {class_name: probability}
        """
        query_tensor = self.transform(query_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(query_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Формируем результат
        class_probs = {}
        for idx, prob in enumerate(probabilities):
            class_name = self.idx_to_label[idx]
            class_probs[class_name] = float(prob)
        
        return class_probs 