import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
import torch.nn.functional as F
from torchsummary import summary
import sys



class DenseNet121(nn.Module):

    def __init__(self, args):
        super(DenseNet121, self).__init__()
        img_model = models.densenet121(pretrained = True)
        self.args = args
        img_model.classifier = torch.nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1024, args.num_classes), nn.Sigmoid()
        )       
        
        self.img_model = img_model
    
    def forward(self, x):
        x = self.img_model(x)
        return(x)