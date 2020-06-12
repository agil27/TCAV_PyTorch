import torch.nn as nn
from torchvision import models


class Resnet18(nn.Module):
    '''simple resnet classifier'''

    def __init__(self, output_num=31):
        super(Resnet18, self).__init__()
        model = models.resnet18(pretrained=False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3)
        self.avgpool = model.avgpool
        self._in_features = model.fc.in_features
        self.cls_layer = nn.Linear(self._in_features, output_num)
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.fill_(0.0)

    def forward(self, x):
        feature = self.feature_layers(x)
        a = self.layer4(feature)
        b = self.avgpool(a)
        c = b.view(b.size(0), -1)
        y = self.cls_layer(c)
        return y
