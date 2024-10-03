import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import torch.nn.functional as F
from torchsummary import summary

import sys


class ResNet50(nn.Module):

    def __init__(self, args):
        super(ResNet50, self).__init__()
        img_model = models.resnet50(pretrained = True)
        self.args = args
        img_model.fc = torch.nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, args.num_classes), nn.Sigmoid()
        )       
        
        self.img_model = img_model
    
    def forward(self, x):
        x = self.img_model(x)
        return(x)