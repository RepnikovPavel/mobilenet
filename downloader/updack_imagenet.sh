#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Использование: $0 /path/to/dir"
  exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
  echo "Ошибка: директория '$DIR' не существует"
  exit 1
fi

shopt -s nullglob
cd "$DIR"

# Рекурсивная функция распаковки
unpack_recursive() {
  local target_dir="$1"
  cd "$target_dir"
  
  # Обрабатываем архивы в текущей директории
  for f in *.tar *.tar.gz *.tgz *.tar.xz *.tar.zst; do
    [ -e "$f" ] || continue
    
    echo "Распаковка: $f"
    
    # Создаем директорию с именем архива (без расширения)
    local subdir="${f%.*}"
    if [[ "$f" == *.tar.* ]]; then
      subdir="${f%.*.*}"
    fi
    
    mkdir -p "$subdir"
    
    case "$f" in
      *.tar.gz|*.tgz)
        tar -xzf "$f" -C "$subdir"
        ;;
      *.tar.xz)
        tar -xJf "$f" -C "$subdir"
        ;;
      *.tar.zst)
        tar --zstd -xf "$f" -C "$subdir"
        ;;
      *.tar)
        tar -xf "$f" -C "$subdir"
        ;;
      *)
        echo "Неизвестный формат: $f, пропуск"
        continue
        ;;
    esac
    
    rm -f "$f"
    
    # Рекурсивно распаковываем содержимое новой папки
    unpack_recursive "$subdir"
  done
  
  cd ..
}

# Запускаем рекурсивную распаковку для каждого основного архива
for f in *.tar *.tar.gz *.tgz *.tar.xz *.tar.zst; do
  [ -e "$f" ] || continue
  unpack_recursive .
done
