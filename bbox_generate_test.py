# 计算 CAM 图，生成 heatmap，并根据 heatmap 计算其各个 disease 对应的 bbox
# 生成 bbox 后选取 CAM 值显著的两个 bbox 记录并存储到 pkl 文件。
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CAM method
methods = {
    "gradcam": GradCAM,
}
# Disease labels
disease_categories = ["Atelectasis", "Cardiomegaly", "Effusion",
                      "Infiltrate", "Mass", "Nodule", "Pneumonia",
                      "Pneumothorax", "Consolidation", "Edema",
                      "Emphysema", "Fibrosis", "Pleural_Thickening",
                      "Hernia"]
            
img_ratio = int(1024/args.img_size)
# 加载 pth 文件
model_dict = torch.load("SavedModels/resnet50_True_MAX_IMG_SIZE_512_num_class_14_epochs_10_best_model.pth")
model_dict = model_dict["model_state_dict"]
model = ResNet50(args).to(device)
model.load_state_dict(model_dict)
model.eval()
target_layers = [model.img_model.layer4[-1]]
# densenet121 取 features[-1] 层
#target_layers = [model.img_model.features[-1]]

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

img_bbox_dict = {}

# 测试接口, 选取 img_name_list 部分图像, 
#img_name_list = img_name_list[0:500]
#img_label_list = img_label_list[0:500]
for i in range(len(img_name_list)):
    img_original = Image.open(os.path.join(data_dir, "images", img_name_list[i])).convert("L")
    img = img_original.resize((args.img_size, args.img_size))
    img = np.array(img)
    img = np.stack((img,) * 3, axis=-1)
    img = np.float32(img) / 255
    # print(img.shape)
    #input_tensor = torch.from_numpy(img)
    input_tensor = preprocess_image(img)
    # 获得各个 class 的 prediction probability
    predicted_tensor = model(input_tensor.to(device))
    predicted_tensor = predicted_tensor[0]
    # 获得预测的 label
    if predicted_tensor.max() >= 0.5:
        predicted_label = torch.argmax(predicted_tensor)
        predicted_label = disease_categories[int(predicted_label.cpu())]
    else:
        predicted_label = "None"
    #predicted_label = get_label(disease_categories, predicted_label)
    
    # 获得 GT label
    gt_labels = []
    gt_labels_text = ""
    for j in range(len(disease_categories)):
        if img_label_list[i][j] == str(1):
            gt_labels.append(disease_categories[j])
            gt_labels_text += disease_categories[j] + "_"
    if gt_labels_text == "":
        gt_labels_text = "None"
    # 生成 CAM
    class_index = 0
    # targets = None
    targets = [ClassifierOutputTarget(class_index)]
    cam_algorithm = methods["gradcam"]
    with cam_algorithm(model = model, target_layers = target_layers, use_cuda = True) as cam:
        grayscale_cam = cam(input_tensor = input_tensor,
                            targets = targets,
                            )
        grayscale_cam = grayscale_cam[0, :]

        _, thresh = cv2.threshold(grayscale_cam, 0.4, 1, cv2.THRESH_BINARY)
        cam_image = show_cam_on_image(img, grayscale_cam * 0.8, use_rgb = True)

        contours= cv2.findContours((thresh.astype(np.uint8)), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        bbox_pos = []
        for c_0 in contours:
            area = cv2.contourArea(c_0)
            x, y, w, h = cv2.boundingRect(c_0)
            bbox_pos.append([x,y,w,h])

    # 筛选 cam 值前 2 的 bbox
    # 获得每个 bbox 的 cam max
    cam_max_list = []
    for j in range(len(bbox_pos)):
        x_0, y_0, x_1, y_1 = int(bbox_pos[j][0]), int(bbox_pos[j][1]), int(bbox_pos[j][0] + bbox_pos[j][2]), int(bbox_pos[j][1] + bbox_pos[j][3]), 
        cam_max = grayscale_cam[x_0:x_1, y_0:y_1].max()
        cam_max_list.append(cam_max)
    #print(img_name_list[i], cam_max_list)
    if len(cam_max_list) == 0:
        bbox_pos_2 = [[0, 0, 1, 1], [0, 0, 1, 1]]
    elif len(cam_max_list) == 1:
        bbox_pos_2 = [bbox_pos[0], bbox_pos[0]]
    else:
        index_list = np.argsort(np.array(cam_max_list))
        #print(img_name_list[i], index_list)
        bbox_pos_2 = [bbox_pos[index_list[-1]], bbox_pos[index_list[-2]]]

    img_bbox_dict.update({img_name_list[i]:bbox_pos_2})
    with open("img_bbox_test_{}.pkl".format(disease_categories[class_index]), "wb") as f:  # 获得的图片的 bbox
        pickle.dump(img_bbox_dict, f)

    plt.figure()
    plt.axis("off")
    plt.imshow(cam_image)
    
    fig = plt.gcf()
    ax = plt.gca()
    # 绘制 predicted bbox
    for k in range(len(bbox_pos_2)):
        bbox = bbox_pos_2[k]
        bbox_x,bbox_y,bbox_w,bbox_h = bbox[0], bbox[1], bbox[2], bbox[3]
        ax.add_patch(plt.Rectangle(xy=(bbox_x, bbox_y,),
                                    width=bbox_w, 
                                    height=bbox_h,
                                    edgecolor = "g",
                                    fill=False, linewidth=3))
        ax.text(0, -2, gt_labels_text.replace("_", " "), fontsize = 20, color = "r",
                horizontalalignment = "left",
                verticalalignment='bottom',
                bbox = dict(facecolor = "r", alpha = 0, clip_on=True, edgecolor="r"))

    result_name = img_name_list[i].split(".")[0] + "_" + args.backbone + "_" + disease_categories[class_index] + "_cam_bbox.png"
    fig.savefig(os.path.join("Heatmap_output_Test", result_name))
