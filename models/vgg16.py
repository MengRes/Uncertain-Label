import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import torch.nn.functional as F
from torchsummary import summary

import sys


class VGG16_Net(nn.Module):

    def __init__(self, args):
        super(VGG16_Net, self).__init__()
        img_model = models.vgg16(pretrained = True)
        img_model.classifier = torch.nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 8),
            nn.Sigmoid()
        )
        self.args = args
        self.img_model = img_model
    
    def forward(self, x):
        x = self.img_model(x)
        return(x)