#!/bin/bash
# download_coco_full.sh — полный COCO 2017 с разметкой (instance + panoptic + stuff)
# Общий размер: ~48 GB
# Использование: ./download_coco_full.sh /path/to/coco

set -e  # Остановка при ошибке

COCO_DIR="${1:-./coco}"
mkdir -p "$COCO_DIR" && cd "$COCO_DIR"

echo "📥 Скачиваю полный COCO 2017 (~48 GB) в $COCO_DIR..."

# 1. ИЗОБРАЖЕНИЯ (25 GB)
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

# 2. ОСНОВНАЯ РАЗМЕТКА (241 MB) — instances (object detection + instance seg)
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 3. PANOPTIC РАЗМЕТКА (821 MB) — things + stuff
wget -c http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

# 4. STUFF РАЗМЕТКА (1.1 GB) — semantic segmentation (фон/поверхности)
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

# 5. INFO для test/unlabeled (опционально, 5 MB)
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

echo "🔓 Распаковываю..."

# РАСПАКОВКА
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip
unzip -q annotations_trainval2017.zip
unzip -q panoptic_annotations_trainval2017.zip
unzip -q stuff_annotations_trainval2017.zip
unzip -q image_info_test2017.zip
unzip -q image_info_unlabeled2017.zip

# ОЧИСТКА
rm *.zip

echo "✅ Готово! Структура:"
tree annotations/ -L 2 || find annotations -maxdepth 2 -type d

echo "
📊 ИТОГОГ SKIPPED (нет разметки):
├── train2017/ (118K img)
├── val2017/   (5K img) 
├── test2017/  (41K img)
└── annotations/
    ├── instances_train2017.json     ← Instance segmentation
    ├── instances_val2017.json       ← Instance segmentation
    ├── person_keypoints_train2017.json
    ├── stuff_train2017.json         ← Stuff segmentation
    ├── stuff_val2017.json
    ├── panoptic_train2017/          ← Panoptic (things+stuff)
    ├── panoptic_val2017/
    └── ..."
