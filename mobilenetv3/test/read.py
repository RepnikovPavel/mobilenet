from mobilenetv3 import mobilenetv3_large
from mobilenetv3 import mobilenetv3_small
import torch

if __name__ == '__main__':
    print('start init')
    net_large = mobilenetv3_large()
    net_small = mobilenetv3_small()
    print('end init')
    print('start reading ckpt')
    net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
    net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))
    print('end reading ckpt')

