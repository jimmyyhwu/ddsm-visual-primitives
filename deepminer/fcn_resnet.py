import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

class FcnResNet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(x)

        return x

    def surgery(self):
        self.cpu()
        num_classes = self.fc.out_features
        state_dict = self.state_dict()
        state_dict['fc.weight'] = state_dict['fc.weight'].view(num_classes, 2048, 1, 1)
        self.fc = nn.Conv2d(2048, num_classes, kernel_size=(1, 1))
        self.load_state_dict(state_dict)
        self.cuda()

def resnet152(**kwargs):
    return FcnResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
