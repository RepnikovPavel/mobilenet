#!/bin/bash
# download_coco_full.sh — полный COCO 2017 с разметкой (instance + panoptic + stuff)
# Общий размер: ~48 GB
# Использование: ./download_coco_full.sh /path/to/coco

set -e  # Остановка при ошибке

COCO_DIR="${1:-/mnt/nvme/datasets/COCO2017/}"
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

