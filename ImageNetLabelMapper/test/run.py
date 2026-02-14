from ImageNetLabelMapper import ImageNetLabelMapper

if __name__ == "__main__":
    classes_file = '/mnt/nvme/datasets/ImageNetLSVRC2012/LOC_synset_mapping.txt'
    
    mapper = ImageNetLabelMapper(classes_file)
    
    synset_id = 'n01819313'
    
    label = mapper.get_label_from_synset(synset_id)
    print(f"{synset_id} → label: {label}")
    
    back_synset = mapper.get_synset_from_label(label)
    print(f"label {label} → {back_synset}")
    
    name = mapper.get_name_from_synset(synset_id)
    print(f"{synset_id} → '{name}'")
    
    name_from_label = mapper.get_name_from_label(label)
    print(f"label {label} → '{name_from_label}'")
    
    print("\n=== XML ПАРСЕР ===")
    info = mapper.get_class_info('n01440764')
    print(f"n01440764 → {info}")
