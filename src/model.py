import torch.nn as nn
import torchvision.models as models


class ImgClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ImgClassifier, self).__init__()
        # load resnet
        # The original paper, The iMaterialist Fashion Attribute Dataset, used ResNet101
        resnet = models.resnet152(pretrained=pretrained)
        # reinitialize the last fully connected layer
        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#resnet
        resnet.fc = nn.Linear(2048, 228)
        self.resnet = resnet

    def forward(self, x):
        out = self.resnet(x)
        return out

    def predict(self, x):
        out = self.forward(x)
        out.sigmoid_()
        return out
