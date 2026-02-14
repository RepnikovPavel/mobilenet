"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    
    # Print warning if input is not divisible by 8
    if v % 8 != 0:
        print(f"Warning: input value {v} is not divisible by 8, adjusted to {new_v}")
    
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        self.width_mult=width_mult
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def count_parameters(self, trainable_only: bool = True) -> float:
        """
        Подсчитывает общее число параметров модели в миллионах
        
        Args:
            trainable_only: считать только обучаемые параметры (requires_grad=True)
        
        Returns:
            float: число параметров в миллионах (M)
        """
        if trainable_only:
            params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            params = sum(p.numel() for p in self.parameters())
        
        return params
    
    def _calculate_madd(self, input_size=224):
        """
        Calculate total multiply-add operations for the network.
        
        Analyzes the structure of the initialized model (self.features, self.conv, self.classifier)
        to determine dimensions and operations, avoiding hardcoding of configs.
        
        Returns:
            total_madd: Total number of MAdd operations
            details: List of per-layer MAdd breakdown
        """
        total_madd = 0
        details = []
        
        h = w = input_size
        c = 3  # Initial input channels (RGB)
        
        # Helper function for Conv2d MAdd calculation
        def count_conv(m, in_c, in_h, in_w):
            # Calculate output dimensions
            # Formula: H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
            out_h = (in_h + 2*m.padding[0] - m.dilation[0]*(m.kernel_size[0]-1) - 1)//m.stride[0] + 1
            out_w = (in_w + 2*m.padding[1] - m.dilation[1]*(m.kernel_size[1]-1) - 1)//m.stride[1] + 1
            
            # MAdd = out_elements * ops_per_element
            # ops_per_element = kernel_h * kernel_w * (in_channels / groups)
            kernel_ops = m.kernel_size[0] * m.kernel_size[1]
            in_channels_per_group = in_c // m.groups
            
            madd = out_h * out_w * m.out_channels * kernel_ops * in_channels_per_group
            return madd, m.out_channels, out_h, out_w

        # Helper function for Linear MAdd calculation
        def count_linear(m):
            # MAdd = in_features * out_features
            return m.in_features * m.out_features

        # 1. Analyze self.features
        # self.features is an nn.Sequential starting with the stem convolution
        # followed by InvertedResidual blocks.
        
        # Process Stem (Index 0 of self.features)
        stem = self.features[0]
        # stem is a Sequential: Conv2d -> BN -> h_swish
        # We only count the Conv2d
        conv_layer = stem[0]
        madd, c, h, w = count_conv(conv_layer, c, h, w)
        total_madd += madd
        details.append({'name': 'Stem', 'madd_M': madd/10**6, 'size': (h, w), 'channels': c})
        
        # Process InvertedResidual blocks (Index 1 onwards)
        for i, block in enumerate(self.features[1:]):
            block_madd = 0
            
            # Temporary state for the block's internal flow
            curr_c = c
            curr_h = h
            curr_w = w
            
            # Inspect block.conv (nn.Sequential)
            # It contains layers like: Conv, BN, Act, SE, Conv, BN...
            for sub_module in block.conv:
                if isinstance(sub_module, nn.Conv2d):
                    m, curr_c, curr_h, curr_w = count_conv(sub_module, curr_c, curr_h, curr_w)
                    block_madd += m
                elif isinstance(sub_module, SELayer):
                    # SE Layer: Global Pool -> FC -> ReLU -> FC -> Hsigmoid
                    # We count the two Linear layers inside self.fc
                    # Input to SE is curr_c channels
                    fc_seq = sub_module.fc
                    # fc_seq structure: Linear -> ReLU -> Linear -> Hsigmoid
                    
                    # First Linear
                    l1 = fc_seq[0]
                    m1 = count_linear(l1)
                    
                    # Second Linear
                    l2 = fc_seq[2]
                    m2 = count_linear(l2)
                    
                    # SE operations apply to all spatial positions (1x1 after pool),
                    # but usually calculated just for the vector.
                    # If we assume the SE vector is broadcasted back, the broadcast multiply
                    # is element-wise (curr_h * curr_w * curr_c), which is often negligible
                    # compared to convolutions or counted as simple ops.
                    # Standard practice is to count the FC MAdds.
                    block_madd += (m1 + m2)
                    
            # Check for residual connection add
            # block.identity is True if stride=1 and inp==oup
            # The add operation is curr_h * curr_w * curr_c (element-wise add)
            if block.identity:
                block_madd += curr_h * curr_w * curr_c
            
            total_madd += block_madd
            details.append({
                'name': f'Block_{i+1}',
                'madd_M': madd/10**6,
                'size': (curr_h, curr_w),
                'channels': curr_c
            })
            
            # Update global state for next block
            c = curr_c
            h = curr_h
            w = curr_w

        # 2. Analyze self.conv (Final 1x1 conv)
        # self.conv is a Sequential: Conv -> BN -> h_swish
        final_conv = self.conv[0]
        madd, c, h, w = count_conv(final_conv, c, h, w)
        total_madd += madd
        details.append({'name': 'Final_Conv', 'madd_M': madd/10**6, 'size': (h, w), 'channels': c})

        # 3. Analyze self.avgpool
        # AdaptiveAvgPool2d reduces dimensions to 1x1.
        h, w = 1, 1

        # 4. Analyze self.classifier
        # Sequential: Linear -> h_swish -> Dropout -> Linear
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                madd = count_linear(layer)
                total_madd += madd
                details.append({
                    'name': f'Linear_{layer.out_features}',
                    'madd_M': madd/10**6,
                    'in_f': layer.in_features,
                    'out_f': layer.out_features
                })
                c = layer.out_features

        return total_madd, details
    


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)

