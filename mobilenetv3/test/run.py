from mobilenetv3 import mobilenetv3_large
import torch
from PIL import Image
from torchvision import transforms
from ImageNetLabelMapper import ImageNetLabelMapper
import cv2

if __name__ == '__main__':
    device = 'cuda:0'
    model = mobilenetv3_large()
    model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
    model.to(device)
    model.eval()

    class_mapper = ImageNetLabelMapper('/mnt/nvme/datasets/ImageNetLSVRC2012/LOC_synset_mapping.txt')
    img_path = '/mnt/nvme/datasets/ImageNetLSVRC2012/ILSVRC2012_img_val/n02106550/ILSVRC2012_val_00033334.JPEG'

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # numpy RGB → PIL RGB
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img_rgb).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(1).item()
        
        # Top-5
        top5_probs, top5_indices = torch.topk(torch.softmax(output, 1), 5)

    print(f"MobileNetV3 предсказание: {pred_idx}")
    print(f"Ожидаемый synset: {class_mapper.get_synset_from_label(pred_idx)}")
    print(f"Название: {class_mapper.get_name_from_label(pred_idx)}")
    print("\nTop-5:")
    for i, idx in enumerate(top5_indices[0].cpu().numpy()):
        prob = top5_probs[0, i].item()
        synset = class_mapper.get_synset_from_label(idx)
        name = class_mapper.get_name_from_label(idx)
        print(f"  {i+1}. {idx}: {name} ({prob:.3f})")