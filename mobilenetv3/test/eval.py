#!/usr/bin/env python3
"""
Оценка MobileNetV3 на ImageNet - расчёт Top-1 и Top-5 Accuracy

Использование:
    python evaluate_mobilenetv3.py --data /path/to/ILSVRC2012_img_val --model large

Требования:
    pip install torch torchvision tqdm
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from mobilenetv3 import mobilenetv3_large
from mobilenetv3 import mobilenetv3_small


# =============================================================================
# Evaluation Functions
# =============================================================================

def get_val_transforms():
    """
    Стандартные трансформации для валидации ImageNet по статье MobileNetV3
    
    В статье используется:
    - Resize до 256 пикселей по короткой стороне
    - Center crop до 224x224
    - Нормализация с ImageNet статистиками
    """
    return transforms.Compose([
        transforms.Resize(256),                    # Resize короткой стороны до 256
        transforms.CenterCrop(224),                # Center crop 224x224
        transforms.ToTensor(),                     # Конвертация в tensor [0, 1]
        transforms.Normalize(                      # ImageNet нормализация
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def prepare_val_dataset(data_path):
    """
    Подготовка validation dataset ImageNet
    
    ВАЖНО: Структура папки должна быть:
    ILSVRC2012_img_val/
        n01440764/
            ILSVRC2012_val_00000293.JPEG
            ...
        n01443537/
            ...
    
    Если изображения лежат в одной папке без подпапок классов,
    нужно сначала запустить скрипт подготовки (см. prepare_imagenet_val.py)
    """
    val_transform = get_val_transforms()
    
    dataset = datasets.ImageFolder(
        root=data_path,
        transform=val_transform
    )
    
    return dataset



def evaluate_topk(model, dataloader, device, k=(1, 5)):
    """
    Вычисление Top-k accuracy
    
    Args:
        model: Модель PyTorch
        dataloader: DataLoader с валидационными данными
        device: CPU или CUDA
        k: Кортеж значений k для top-k accuracy (по умолчанию (1, 5))
    
    Returns:
        top1_acc, top5_acc в процентах
    """
    model.eval()
    
    topk_correct = [0] * len(k)
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Вычисляем top-k predictions
            # outputs: [batch_size, num_classes]
            # targets: [batch_size]
            
            batch_size = targets.size(0)
            total += batch_size
            
            # Получаем top-k индексы для каждого примера
            _, predictions = outputs.topk(max(k), dim=1)  # [batch_size, max_k]
            
            # Сравниваем с истинными метками
            # predictions.t() транспонирует для удобного сравнения
            correct = predictions.eq(targets.view(-1, 1).expand_as(predictions))
            
            # Считаем количество правильных для каждого k
            for i, ki in enumerate(k):
                topk_correct[i] += correct[:, :ki].sum().item()
    
    # Вычисляем accuracy в процентах
    topk_acc = [100.0 * correct / total for correct in topk_correct]
    
    return topk_acc


def count_parameters(model):
    """Подсчёт количества параметров модели"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_runs=100):
    """Измерение времени инференса"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Измерение
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_runs * 1000  # в миллисекундах
    
    return avg_time


def load_pretrained_weights(model, model_type, pretrained_path=None):
    """
    Загрузка предобученных весов
    
    Варианты:
    1. Указать путь к файлу весов через --weights
    2. Использовать torchvision pretrained модели
    3. Инициализировать случайно (для отладки)
    """
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Загрузка весов из: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("ВНИМАНИЕ: Используются случайно инициализированные веса!")
        print("Для реальной оценки загрузите предобученные веса через --weights")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Оценка MobileNetV3 на ImageNet')
    parser.add_argument('--data', type=str, required=True,
                        help='Путь к папке с validation данными ImageNet')
    parser.add_argument('--model', type=str, default='large', choices=['large', 'small'],
                        help='Вариант модели: large или small')
    parser.add_argument('--weights', type=str, default=None,
                        help='Путь к файлу с весами модели')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Размер батча для оценки')
    parser.add_argument('--workers', type=int, default=4,
                        help='Количество workers для DataLoader')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Устройство для вычислений')
    
    args = parser.parse_args()
    
    # Определяем устройство
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA недоступна, переключаемся на CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("ОЦЕНКА MobileNetV3 НА IMAGENET")
    print("=" * 60)
    print(f"Модель: MobileNetV3-{args.model}")
    print(f"Устройство: {device}")
    print(f"Данные: {args.data}")
    print()
    
    # Создаём модель
    print("Создание модели...")
    if args.model == 'large':
        model = mobilenetv3_large(num_classes=1000)
    else:
        model = mobilenetv3_small(num_classes=1000)
    
    # Загружаем веса
    model = load_pretrained_weights(model, args.weights)
    model = model.to(device)
    
    # Информация о модели
    total_params, trainable_params = count_parameters(model)
    print(f"\nПараметры модели:")
    print(f"  Всего: {total_params:,}")
    print(f"  Обучаемых: {trainable_params:,}")
    print(f"  Размер: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Подготавливаем данные
    print("\nЗагрузка данных...")
    try:
        dataset = prepare_val_dataset(args.data)
        print(f"Количество изображений: {len(dataset)}")
        print(f"Количество классов: {len(dataset.classes)}")
    except Exception as e:
        print(f"ОШИБКА при загрузке данных: {e}")
        print("\nУбедитесь, что структура папки правильная:")
        print("  ILSVRC2012_img_val/")
        print("    n01440764/")
        print("      ILSVRC2012_val_00000293.JPEG")
        print("    ...")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Оценка
    print("\nОценка модели...")
    start_time = time.time()
    top1_acc, top5_acc = evaluate_topk(model, dataloader, device, k=(1, 5))
    eval_time = time.time() - start_time
    
    # Измерение времени инференса
    print("\nИзмерение времени инференса...")
    avg_inference_time = measure_inference_time(model, device)
    
    # Результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"\nВремя оценки: {eval_time:.1f} сек")
    print(f"Среднее время инференса: {avg_inference_time:.2f} мс")
    
    # Сравнение с оригинальной статьёй
    print("\n" + "-" * 60)
    print("Сравнение с оригинальной статьёй MobileNetV3:")
    print("-" * 60)
    
    reference = {
        'large': {'top1': 75.2, 'top5': 92.5, 'params': 5.4},
        'small': {'top1': 67.4, 'top5': 87.1, 'params': 2.5}
    }
    
    ref = reference[args.model]
    print(f"Статья: Top-1={ref['top1']}%, Top-5={ref['top5']}%, Params={ref['params']}M")
    print(f"Ваши:   Top-1={top1_acc:.2f}%, Top-5={top5_acc:.2f}%, Params={total_params/1e6:.2f}M")
    
    if args.weights is None:
        print("\n⚠️  ВНИМАНИЕ: Результаты низкие, т.к. используются случайные веса!")
        print("   Загрузите предобученные веса через --weights path/to/weights.pth")


if __name__ == '__main__':
    main()