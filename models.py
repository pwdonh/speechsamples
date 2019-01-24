import torchvision.models as models
import torch.nn as nn
import math
import torch.nn.functional as F
import torch

# adapted from the torch resnet
class VoxResNet(models.resnet.ResNet):
    def __init__(self, block, layers, embed_size, num_classes=5995):
        self.inplanes = 64
        super(VoxResNet, self).__init__(block, layers, num_classes)
        # 1 channel except 3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # fully connected layers to be invariant to temporal position
        self.fcpre = nn.Conv2d(512, 512, kernel_size=(9,1))
        # self.avgpool = nn.AvgPool2d((1,10))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_mu = nn.Linear(512, embed_size)
        self.fc = nn.Linear(embed_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if embed_size==512:
            self.fc_mu.weight.data = torch.eye(512)
            self.fc_mu.bias.data.zero_()

        self.loss = nn.CrossEntropyLoss()
        self.loss_eval = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.trunk(x)
        x = self.fc_mu(x)
        x = self.fc(x)

        return x

    def trunk(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fcpre(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class VoxResNetVAE(VoxResNet):
    def __init__(self, block, layers, embed_size, num_classes=5995):
        super(VoxResNetVAE, self).__init__(block, layers, embed_size, num_classes)
        self.fc_var = nn.Linear(512, embed_size)
        self.loss = LossVAE()

    def encode(self, x):
        x = self.trunk(x)
        return self.fc_mu(x), self.fc_var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        z, logvar = self.encode(x)
        if self.training:
            z = self.reparameterize(z, logvar)
        x = self.fc(z)
        return x, z, logvar

    def loss_eval(self, input, target):
        input, mu, logvar = input
        return F.cross_entropy(input, target)

class LossVAE(nn.Module):
    def __init__(self):
        super(LossVAE, self).__init__()

    def forward(self, input, target):
        input, mu, logvar = input
        CE = F.cross_entropy(input, target)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return CE, KLD

def voxresnet34(model_type=VoxResNet, embed_size=512, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if model_type=='VoxResNet':
        model = VoxResNet(models.resnet.BasicBlock, [3, 4, 6, 3], embed_size, **kwargs)
    elif model_type=='VoxResNetVAE':
        model = VoxResNetVAE(models.resnet.BasicBlock, [3, 4, 6, 3], embed_size, **kwargs)
    return model
