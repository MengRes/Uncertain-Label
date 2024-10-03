import os
import sys
import numpy as np
import cv2
from PIL import Image

import argparse
from config import parser
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50
from models.densenet121 import DenseNet121
from models.vgg16 import VGG16_Net
import matplotlib.pyplot as plt
import pickle

import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms

from cam_dicts.grad_cam import GradCAM
from cam_dicts.utils.image import show_cam_on_image
from tool.utils import get_iou, preprocess_image, get_label
from tool.utils import find_max_pos, xy_in_bbox, mask_find_bboxs
from cam_dicts.utils.model_targets import ClassifierOutputTarget
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Disease labels
disease_categories = ["Atelectasis", "Cardiomegaly", "Effusion",
                      "Infiltrate", "Mass", "Nodule", "Pneumonia",
                      "Pneumothorax", "Consolidation", "Edema",
                      "Emphysema", "Fibrosis", "Pleural_Thickening",
                      "Hernia"]

# 加载 bbox 文件
img_bbox_dict_list = []
for j in range(len(disease_categories)):
    img_bbox_dict = open("img_bbox_test/img_bbox_test_{}.pkl".format(disease_categories[j]), "rb")
    data = pickle.load(img_bbox_dict)
    img_bbox_dict_list.append(data)

# 加载 pth 文件
model_dict = torch.load("SavedModels/resnet50_True_MAX_IMG_SIZE_512_num_class_14_epochs_10_best_model.pth")
model_dict = model_dict["model_state_dict"]
model = ResNet50(args).to(device)
model.load_state_dict(model_dict)
model.eval()

# 提取图像 feature
def backward_hook(module, grad_in, grad_out):
    grad_block["grad_in"] = grad_in
    grad_block["grad_out"] = grad_out

def forward_hook(module, inp, outp):
    fmap_block["input"] = inp
    fmap_block["output"] = outp

'''
input:
    img_size: [w0, h0]
    bbox_size: [x1, y1, w1, h1]  
'''

def get_bbox_feature(bbox_size, img_feature, output_size):
    pooler = torchvision.ops.RoIAlign(output_size=output_size, sampling_ratio=2, spatial_scale=1)
    x1, y1 = bbox_size[0], bbox_size[1]
    x2, y2 = bbox_size[0] + bbox_size[2], bbox_size[1] + bbox_size[3]
    box = torch.tensor([[x1, y1, x2, y2]])
    output = pooler(img_feature.cpu(), [box])
    return output

# Loading images
data_dir = os.path.join(args.data_root_dir, "dataset")
data_image_list = []
data_label_list = []
data_img_size = []

# 获取图像及其 label
with open(os.path.join(data_dir, "test_label.csv"), "r") as f:
    lines = f.readlines()
    lines = lines[1:]
    img_name_list = []
    img_label_list = []
    for i in range(len(lines)):
        line = lines[i].split(",")
        # name list
        img_name_list.append(line[0])
        # label list
        img_label_list.append(line[1:15])

img_size = [512, 512]
feature_map_size = [16, 16]
feature_ratio = img_size[0] / feature_map_size[0]
disease_feature_dict_dict = {}

# 测试接口, 选取 img_name_list 部分图像, 
#img_name_list = img_name_list[0:100]
#img_label_list = img_label_list[0:100]
for i in range(len(img_name_list)):
    img_original = Image.open(os.path.join(data_dir, "images", img_name_list[i])).convert("L")
    img = img_original.resize((args.img_size, args.img_size))
    img = np.array(img)
    img = np.stack((img,) * 3, axis=-1)
    img = np.float32(img) / 255

    input_tensor = preprocess_image(img).requires_grad_() 
    fmap_block = dict()  # 装feature map
    #grad_block = dict()  # 装梯度

    # 注册hook
    model.img_model.layer4[-1].register_forward_hook(forward_hook)
    #model.img_model.layer4[-1].register_backward_hook(backward_hook)
    # 获得各个 class 的 prediction probability
    predicted_tensor = model(input_tensor.to(device)).cpu()
    #print(predicted_tensor.shape)      # torch.Size([1, 14])
    #print(fmap_block["input"][0].shape) # torch.Size([1, 2048, 16, 16])
    #print(fmap_block["output"].shape)   # torch.Size([1, 2048, 16, 16])

    disease_bbox_list = []
    disease_feature_dict = {}
    for k in range(len(img_bbox_dict_list)):
        disease_bbox = img_bbox_dict_list[k][img_name_list[i]]
        disease_bbox_list.append(disease_bbox)
        if len(disease_bbox) == 0:
            disease_bbox = [[0, 0, 0, 0]]

        output_feature_list = []
        for m in range(len(disease_bbox)):
            feature_bbox_size = [x/feature_ratio for x in disease_bbox[m]]
            output_feature = get_bbox_feature(bbox_size=feature_bbox_size, img_feature=fmap_block["output"], output_size=1)
            output_feature = output_feature.detach().numpy()
            output_feature_list.append(output_feature)
        disease_feature_dict.update({disease_categories[k]:output_feature_list})
    disease_feature_dict_dict.update({img_name_list[i]:disease_feature_dict})
    with open("img_disease_feature_test.pkl", "wb") as f:
        pickle.dump(disease_feature_dict_dict, f)
    input_tensor.detach()
    fmap_block["output"].detach()
    torch.cuda.empty_cache()



    

        



    
