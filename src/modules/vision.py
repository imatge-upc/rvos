from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.vgg import VGG, make_layers
import torch.nn as nn
import math

class ResNet101(ResNet):
    """Returns intermediate features from ResNet-50"""
    def __init__(self):
        super(ResNet101,self).__init__(Bottleneck, [3, 4, 23, 3], 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5,x4,x3,x2,x1

class ResNet50(ResNet):
    """Returns intermediate features from ResNet-50"""
    def __init__(self):
        super(ResNet50,self).__init__(Bottleneck, [3, 4, 6, 3], 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5,x4,x3,x2,x1

class ResNet34(ResNet):
    """Returns intermediate features from ResNet-34"""
    def __init__(self):
        super(ResNet34,self).__init__(BasicBlock, [3, 4, 6, 3], 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x5,x4,x3,x2,x1

class VGG16(nn.Module):

    def __init__(self, num_classes = 1000, features  =  [64, 64, 'M',
                                                       128, 128, 'M',
                                                       256, 256, 256, 'M',
                                                       512, 512, 512, 'M',
                                                       512, 512, 512, 'M']):
        super(VGG16, self).__init__()
        self.features = make_layers(features)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x1 = self.features[4](x)

        x = self.features[5](x1)
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.features[8](x)
        x2 = self.features[9](x)

        x = self.features[10](x2)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x3 = self.features[16](x)

        x = self.features[17](x3)
        x = self.features[18](x)
        x = self.features[19](x)
        x = self.features[20](x)
        x = self.features[21](x)
        x = self.features[22](x)
        x4 = self.features[23](x)

        x = self.features[24](x4)
        x = self.features[25](x)
        x = self.features[26](x)
        x = self.features[27](x)
        x = self.features[28](x)
        x = self.features[29](x)
        x5 = self.features[30](x)

        return x5, x4, x3, x2, x1

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()