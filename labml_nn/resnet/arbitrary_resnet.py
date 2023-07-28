import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck


def make_layer(block, in_channels, channels, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)  # stride for the first block and 1 for the remaining
    layers = []
    for stride in strides:
        layers.append(block(in_channels, channels, stride=stride))
        in_channels = block.expansion*channels
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self._make_layers(block, num_blocks)
        self.linear = nn.Linear(self.in_channels, num_classes)

    def _make_layers(self, block, num_blocks):
        layers = []
        channels = 64
        for num_block in num_blocks:
            layers.append(make_layer(block, self.in_channels, channels, num_block, stride=2))
            self.in_channels = channels * block.expansion
            channels *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
