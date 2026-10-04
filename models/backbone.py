import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, ResNet34_Weights


class BinauralResNet18(nn.Module):
    def __init__(self, in_channel=2, pretrained=False):
        super(BinauralResNet18, self).__init__()
        weight = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weight)
        self.backbone.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=(4, 4), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x_1 = self.backbone.layer1(x)
        x_2 = self.backbone.layer2(x_1)
        x_3 = self.backbone.layer3(x_2)
        x_4 = self.backbone.layer4(x_3)

        return [x_1, x_2, x_3, x_4]


class BinauralResNet34(nn.Module):
    def __init__(self, in_channel=2, pretrained=False):
        super(BinauralResNet34, self).__init__()
        weight = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weight)
        self.backbone.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=(4, 4), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x_1 = self.backbone.layer1(x)
        x_2 = self.backbone.layer2(x_1)
        x_3 = self.backbone.layer3(x_2)
        x_4 = self.backbone.layer4(x_3)

        return [x_1, x_2, x_3, x_4]
