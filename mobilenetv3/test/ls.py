from mobilenetv3 import mobilenetv3_large
import torch

if __name__ == '__main__':
    net_large = mobilenetv3_large()
    state_dict = torch.load('pretrained/mobilenetv3-large-1cd25616.pth')
    net_large.load_state_dict(state_dict)
    for k in state_dict:
        print(f'{k} {state_dict[k].size()}')
