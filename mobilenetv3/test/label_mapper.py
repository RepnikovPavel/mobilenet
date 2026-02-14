import os
import json
from typing import Dict, Union

class ImageNetLabelMapper:
    def __init__(self, classes_file: str):
        self.classes_file = classes_file
        self.synset_to_name_dict = {}  # ← переименовал
        self.synset_to_short_dict = {}
        self.synset_to_idx = {}
        self.idx_to_synset = {}
        self.idx_to_name = {}
        self.name_to_synset = {}
        
        self._load_mapping()
    
    def _load_mapping(self):
        with open(self.classes_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"⚠️  Пропускаю строку {line_num}: '{line}'")
                    continue
                
                synset_id = parts[0].strip()
                class_name = parts[1].strip()
                
                self.synset_to_name_dict[synset_id] = class_name
                short_name = class_name.split(',')[0].strip()
                self.synset_to_short_dict[synset_id] = short_name
                self.name_to_synset[short_name] = synset_id
        
        # Сортируем synset'ы по алфавиту
        sorted_synsets = sorted(self.synset_to_name_dict.keys())
        
        for idx, synset_id in enumerate(sorted_synsets):
            self.synset_to_idx[synset_id] = idx
            self.idx_to_synset[idx] = synset_id
            self.idx_to_name[idx] = self.synset_to_name_dict[synset_id]
        
        print(f"✅ Загружено {len(self.synset_to_name_dict)} классов")
    
    def get_label_from_synset(self, synset_id: str) -> int:
        """synset_id → метка (0-999)"""
        return self.synset_to_idx.get(synset_id, -1)
    
    def get_synset_from_label(self, label: int) -> str:
        """метка → synset_id"""
        if 0 <= label < len(self.idx_to_synset):
            return self.idx_to_synset[label]
        return None
    
    def get_name_from_synset(self, synset_id: str) -> str:  # ← переименовал
        """synset_id → название"""
        return self.synset_to_name_dict.get(synset_id, f"unknown_{synset_id}")
    
    def get_synset_from_name(self, class_name: str) -> str:
        """имя → synset_id"""
        return self.name_to_synset.get(class_name, None)
    
    def get_name_from_label(self, label: int) -> str:
        """метка → название"""
        if 0 <= label < len(self.idx_to_name):
            return self.idx_to_name[label]
        return f"unknown_{label}"
    
    def get_class_info(self, synset_id: str) -> dict:
        """synset → вся информация"""
        label = self.get_label_from_synset(synset_id)
        name = self.get_name_from_synset(synset_id)
        short_name = self.synset_to_short_dict.get(synset_id, "")
        return {
            'label': label,
            'name': name,
            'short_name': short_name
        }

# === ТЕСТ ===
if __name__ == "__main__":
    classes_file = '/mnt/nvme/datasets/ImageNetLSVRC2012/LOC_synset_mapping.txt'
    
    mapper = ImageNetLabelMapper(classes_file)
    
    synset_id = 'n01819313'
    
    # Все направления работают
    label = mapper.get_label_from_synset(synset_id)
    print(f"{synset_id} → label: {label}")
    
    back_synset = mapper.get_synset_from_label(label)
    print(f"label {label} → {back_synset}")
    
    name = mapper.get_name_from_synset(synset_id)  # ← теперь работает!
    print(f"{synset_id} → '{name}'")
    
    name_from_label = mapper.get_name_from_label(label)
    print(f"label {label} → '{name_from_label}'")
    
    print("\n=== XML ПАРСЕР ===")
    info = mapper.get_class_info('n01440764')
    print(f"n01440764 → {info}")
