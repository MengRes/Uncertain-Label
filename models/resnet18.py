import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import torch.nn.functional as F
from torchsummary import summary

import sys


class ResNet18(nn.Module):

    def __init__(self, args):
        super(ResNet18, self).__init__()
        img_model = models.resnet18(pretrained = True)
        self.args = args
        img_model.fc = torch.nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, args.num_classes), nn.Sigmoid()
        )       
        
        self.img_model = img_model
    
    def forward(self, x):
        x = self.img_model(x)
        return(x)