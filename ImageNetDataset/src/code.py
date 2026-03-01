import os
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2


class ImageNetLabelMapper:
    def __init__(self, classes_file: str):
        self.classes_file = classes_file
        self.synset_to_name_dict = {}
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
    def __len__(self):
        return len(self.idx_to_synset)
    def __getitem__(self, intidx):
        return self.get_synset_from_label(intidx)

class ImageNetCls(Dataset):
    train:bool=True
    def __init__(
        self,
        root='/mnt/nvme/datasets/ImageNetLSVRC2012'
    ):
        super().__init__()
        if not os.path.exists(os.path.join(root,'LOC_synset_mapping.txt')):
            raise FileNotFoundError(os.path.join(root,'LOC_synset_mapping.txt'))
        self.label_mapper = ImageNetLabelMapper(os.path.join(root,'LOC_synset_mapping.txt'))
        self.box_train_root = os.path.join(root,'ILSVRC2012_bbox_train_v2')
        self.box_val_root = os.path.join(root,'ILSVRC2012_bbox_val_v3','val')
        self.img_train_root = os.path.join(root,'ILSVRC2012_img_train')
        self.img_val_root = os.path.join(root,'ILSVRC2012_img_val')

        for cls_idx in tqdm(range(len(self.label_mapper)),desc='folder structure validation'):
            cls_id = self.label_mapper[cls_idx]
            
            if not os.path.exists(os.path.join(self.box_train_root,str(cls_id))):
                raise FileNotFoundError(f"invalid folder structure {os.path.join(self.box_train_root,str(cls_id))}")
            if not os.path.exists(os.path.join(self.box_val_root,str(cls_id))):
                raise FileNotFoundError(f"invalid folder structure {os.path.join(self.box_val_root,str(cls_id))}")
            if not os.path.exists(os.path.join(self.img_train_root,str(cls_id))):
                raise FileNotFoundError(f"invalid folder structure {os.path.join(self.img_train_root,str(cls_id))}")
            if not os.path.exists(os.path.join(self.img_val_root,str(cls_id))):
                raise FileNotFoundError(f"invalid folder structure {os.path.join(self.img_val_root,str(cls_id))}")
        
        self.train_img_files = {}
        self.train_box_files = {}

        self.val_img_files = {}
        self.val_box_files = {}
        
        for cls_idx in tqdm(range(len(self.label_mapper)),desc='all files indexing'):
            cls_id = self.label_mapper[cls_idx]
            if cls_id not in self.train_img_files:
                self.train_img_files.update({cls_id:{}})
            if cls_id not in self.val_img_files:
                self.val_img_files.update({cls_id:{}})
            if cls_id not in self.train_box_files:
                self.train_box_files.update({cls_id:{}})
            if cls_id not in self.val_box_files:
                self.val_box_files.update({cls_id:{}})
            
            files = os.listdir(os.path.join(self.box_train_root,str(cls_id)))
            indexes = [int(Path(el).stem.split("_")[1]) for el in files]
            argsort_ = np.argsort(indexes)
            files = [files[sorted_idx] for sorted_idx in argsort_]
            files = [os.path.join(self.box_train_root,str(cls_id),el) for el in files]
            indexes = [indexes[sorted_idx] for sorted_idx in argsort_]
            self.train_box_files[cls_id]['files']=files
            self.train_box_files[cls_id]['num_files']=len(indexes)
            self.train_box_files[cls_id]['indexes']=indexes

            files = os.listdir(os.path.join(self.box_val_root,str(cls_id)))
            indexes = [int(Path(el).stem.split("_")[2]) for el in files]
            argsort_ = np.argsort(indexes)
            files = [files[sorted_idx] for sorted_idx in argsort_]
            files = [os.path.join(self.box_val_root,str(cls_id),el) for el in files]
            indexes = [indexes[sorted_idx] for sorted_idx in argsort_]
            self.val_box_files[cls_id]['files']=files
            self.val_box_files[cls_id]['num_files']=len(indexes)
            self.val_box_files[cls_id]['indexes']=indexes

            files = os.listdir(os.path.join(self.img_train_root,str(cls_id)))
            indexes = [int(Path(el).stem.split("_")[1]) for el in files]
            argsort_ = np.argsort(indexes)
            files = [files[sorted_idx] for sorted_idx in argsort_]
            files = [os.path.join(self.img_train_root,str(cls_id),el) for el in files]
            indexes = [indexes[sorted_idx] for sorted_idx in argsort_]
            self.train_img_files[cls_id]['files']=files
            self.train_img_files[cls_id]['num_files']=len(indexes)
            self.train_img_files[cls_id]['indexes']=indexes

            files = os.listdir(os.path.join(self.img_val_root,str(cls_id)))
            indexes = [int(Path(el).stem.split("_")[2]) for el in files]
            argsort_ = np.argsort(indexes)
            files = [files[sorted_idx] for sorted_idx in argsort_]
            files = [os.path.join(self.img_val_root,str(cls_id),el) for el in files]
            indexes = [indexes[sorted_idx] for sorted_idx in argsort_]
            self.val_img_files[cls_id]['files']=files
            self.val_img_files[cls_id]['num_files']=len(indexes)
            self.val_img_files[cls_id]['indexes']=indexes

            # число файлов с размеченными боксами меньше чем самих изображений            
            # if self.train_box_files[cls_id]['num_files'] != self.train_img_files[cls_id]['num_files']:
            #     raise ValueError(f"self.train_box_files[cls_id]['num_files'] {self.train_box_files[cls_id]['num_files']} != self.train_img_files[cls_id]['num_files'] {self.train_img_files[cls_id]['num_files']}")
            # if self.val_box_files[cls_id]['num_files'] != self.val_img_files[cls_id]['num_files']:
            #     raise ValueError(f"self.val_box_files[cls_id]['num_files'] {self.val_box_files[cls_id]['num_files']} != self.val_img_files[cls_id]['num_files'] {self.val_img_files[cls_id]['num_files']}")

        total_train_cls = 0
        total_val_cls = 0
        total_train_box = 0
        total_val_box = 0
        train_cls_lengths = []
        val_cls_lengths = []
        train_box_lengths = []
        val_box_lengths = []

        for cls_idx in tqdm(range(len(self.label_mapper)),desc='all files indexing'):
            cls_id = self.label_mapper[cls_idx]
            total_train_cls += self.train_img_files[cls_id]['num_files']
            total_val_cls += self.val_img_files[cls_id]['num_files']
            total_train_box += self.train_box_files[cls_id]['num_files']
            total_val_box += self.val_box_files[cls_id]['num_files']
            train_cls_lengths.append(self.train_img_files[cls_id]['num_files'])
            val_cls_lengths.append(self.val_img_files[cls_id]['num_files'])
            train_box_lengths.append(self.train_box_files[cls_id]['num_files'])
            val_box_lengths.append(self.val_box_files[cls_id]['num_files'])

        self.train_cls_lengths=train_cls_lengths
        self.val_cls_lengths=val_cls_lengths
        self.train_box_lengths=train_box_lengths
        self.val_box_lengths=val_box_lengths

        self.total_train_cls=total_train_cls
        self.total_val_cls=total_val_cls
        self.total_train_box=total_train_box
        self.total_val_box=total_val_box
        
        self.train_cls_cumsum = np.cumsum(self.train_cls_lengths)
        self.val_cls_cumsum = np.cumsum(self.val_cls_lengths)
        self.train_box_cumsum = np.cumsum(self.train_box_lengths)
        self.val_box_cumsum = np.cumsum(self.val_box_lengths)

        print(f'total_train_cls {total_train_cls}')
        print(f'total_val_cls {total_val_cls}')
        print(f'total_train_box {total_train_box}')
        print(f'total_val_box {total_val_box}')

    def __len__(self):
        if self.train:
            return self.total_train_cls
        else:
            return self.total_val_cls
        
    def train(self):
        self.train = True
    def eval(self):
        self.train = False

    def __getitem__(self, index):
        cls_id, local_idx = self._map_index(index)

        if self.train:
            img_path = self.train_img_files[cls_id]['files'][local_idx]
        else:
            img_path = self.val_img_files[cls_id]['files'][local_idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return img, self.label_mapper.get_label_from_synset(cls_id)

    def class_index_to_class_name(self,idx):
        return self.label_mapper.get_name_from_label(idx)
        
    def _map_index(self, index: int):
        # index — глобальный 0-based индекс из DataLoader

        if self.train:
            cumsums = self.train_cls_cumsum
            lengths = self.train_cls_lengths
            mapper = self.label_mapper
        else:
            cumsums = self.val_cls_cumsum
            lengths = self.val_cls_lengths
            mapper = self.label_mapper

        # проверка на выход за границы
        if index < 0 or index >= cumsums[-1]:
            raise IndexError(index)

        # переводим глобальный 0-based индекс в 1-based для searchsorted
        idx1 = index + 1

        # позиция класса (0-based по self.label_mapper)
        cls_pos = int(np.searchsorted(cumsums, idx1, side="left"))
        cls_id = mapper[cls_pos]

        # сумма до этого класса (кол-во элементов предыдущих классов)
        prev_cum = 0 if cls_pos == 0 else int(cumsums[cls_pos - 1])

        # локальный индекс внутри класса (0-based)
        local_idx = index - prev_cum

        # sanity-check (можно убрать в проде)
        assert 0 <= local_idx < lengths[cls_pos]

        return cls_id, int(local_idx)
    