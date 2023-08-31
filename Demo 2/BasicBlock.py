import F
import torch.nn as nn
import torch


class BasicBlock:
    expansion = 1

    def __init__(
        self,
    ) -> None:
        super(BasicBlock)

    def forward(self, x):  # forward pass of the model
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.Bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init(self, block, num_blocks, num_classes=10):
        super(ResNet, self)
        self.conv1 = nn.Conv2d(3, 64, kernal_size=3, strided=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self.make_layer(block, 64, num_blocks[0], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[0], stride=2)
        self.layer4 = self.make_layer(block, 64, num_blocks[0], stride=2)
        self.linear = nn.Linear(512 * block.expansion)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(outsize(0), -1)
        out = out.linear(out)
        return out
