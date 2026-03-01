from ImageNetDataset import ImageNetCls

if __name__ == '__main__':
    dataset = ImageNetCls()
    dataset.train()
    print(f"dataset train len {len(dataset)}")
    dataset.eval()
    print(f"dataset eval len {len(dataset)}")
    

    