set -e  # Остановка при ошибке

DIR="${1:-/mnt/nvme/datasets/ImageNetLSVRC2012}"
mkdir -p "$DIR" && cd "$DIR"
cd $DIR
wget https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt

