classes_file = '/mnt/nvme/datasets/ImageNetLSVRC2012/LOC_synset_mapping.txt'

synset_to_name = {}  # synset → полное название
synset_to_short = {}  # synset → первое слово

with open(classes_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
            
        # Разделяем по первому пробелу: "n01440764" и остальное
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            print(f"⚠️  Пропускаю некорректную строку {line_num}: '{line}'")
            continue
            
        synset_id = parts[0].strip()      # 'n01440764'
        class_name = parts[1].strip()     # 'tench, Tinca tinca'
        
        # Полное название
        synset_to_name[synset_id] = class_name
        
        # Первое слово (короткое название)
        short_name = class_name.split(',')[0].strip()
        synset_to_short[synset_id] = short_name

print(f"✅ Загружено {len(synset_to_name)} классов")