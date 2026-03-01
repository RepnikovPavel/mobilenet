from ImageNetDataset import ImageNetCls
from tqdm import tqdm

if __name__ == '__main__':
    dataset = ImageNetCls()
    dataset.train()
    print(f"dataset train len {len(dataset)}")
    dataset.eval()
    print(f"dataset eval len {len(dataset)}")
    
    for i in tqdm(range(len(dataset)),desc='iterate over dataset'):
        img,classidx=dataset[i]
        class_name = dataset.class_index_to_class_name(classidx)
        
    