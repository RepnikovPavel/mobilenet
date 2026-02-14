from mobilenetv3 import mobilenetv3_large
import torch
import requests
from PIL import Image
from torchvision import transforms
from ImageNetLabelMapper import ImageNetLabelMapper
from pprint import pprint

if __name__ == '__main__':
    device = 'cuda:0'
    model = mobilenetv3_large()
    model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
    print(f'params {model.count_parameters()/10**6:.2f} M')
    print(f'madd {model._calculate_madd()[0]/10**6:.2f} M')
    print('madd details')
    pprint(model._calculate_madd()[1])
