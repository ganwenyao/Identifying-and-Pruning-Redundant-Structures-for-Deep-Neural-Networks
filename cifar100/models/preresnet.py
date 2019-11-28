from __future__ import absolute_import
import math

import torch
import torch.nn as nn

__all__ = ['preresnet']

"""
preactivation resnet with bottleneck design.
"""

class Projection(nn.Module):
    def __init__(self, downsample):
        super(Projection, self).__init__()
        self.downsample = downsample
    def forward(self, x):
        return self.downsample(x)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            # print(self.downsample)
            identity = self.downsample(x)

        out += identity
        return out

class preresnet(nn.Module):
    def __init__(self, depth=56, dataset='cifar100', cfg=None):
        super(preresnet, self).__init__()
        block = Bottleneck

        if cfg is None:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            cfg = []
            cfg.append([n, True])
            for i in range(n):
                cfg.append([16, 16, 64])
            cfg.append([n, True])
            for i in range(n):
                cfg.append([32, 32, 128])
            cfg.append([n, True])
            for i in range(n):
                cfg.append([64, 64, 256])
        self.cfg = cfg
        n1, flag1 = cfg[0]
        # print(n1)
        # print(cfg[n1+1])
        # print(cfg)
        n2, flag2= cfg[n1+1]
        n3, flag3 = cfg[n1+n2+2]
        cfg1 = cfg[1:n1+1]
        cfg2 = cfg[n1+2:n1+n2+2]
        cfg3 = cfg[n1+n2+3:]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,bias=False)

        self.layer1 = self._make_layer(block, 16, flag1, cfg = cfg1)
        self.layer2 = self._make_layer(block, 64, flag2, cfg = cfg2, stride=2)
        self.layer3 = self._make_layer(block, 128, flag3, cfg = cfg3, stride=2)
        self.bn = nn.BatchNorm2d(cfg[-1][-1])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1][-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1][-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, first_residual, cfg, stride=1):
        downsample = nn.Conv2d(inplanes, cfg[0][2], kernel_size=1, stride=stride, bias=False)
        layers = []
        if first_residual:
            # print('first_residual')
            # print(downsample)
            layers.append(block(inplanes, cfg[0], stride, downsample))
        else:
            layers.append(Projection(downsample))
            layers.append(block(cfg[0][2], cfg[0]))

        for i in range(1, len(cfg)):
            layers.append(block(cfg[i-1][2], cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = preresnet()
    # x = torch.FloatTensor(16, 3, 40, 40)
    # y = model(x)
    # for name, block in model.named_modules():
    #     if isinstance(block, Bottleneck):
    #         print(name, block)
            # for name, module in block.named_children():
            #     print(name,module)
            # print(block.children())
    print(model)
    # for block in model.children():
    #     print(block)
    # block_list = []
    # for m0 in model.modules():
    #     if isinstance(m0, Bottleneck):
    #         block_list.append(m0)
    # for m0 in model.children():
    #     if isinstance(m0, nn.Conv2d):
    #         print('conv1', m0)
    #         conv_first = m0
    #     elif isinstance(m0, nn.BatchNorm2d):
    #         print('BN1', m0)
    #         BN_last = m0
    #     elif isinstance(m0, nn.Linear):
    #         print('linear', m0)
    #         linear_last = m0