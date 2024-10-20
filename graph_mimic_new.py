## 输入图片的 findings 和 anatomy 的 bbox，绘制 graph 的 node
## 输入 findings 和 anatomy 的关系矩阵， 获得 edge
import os
import numpy as np
import pandas
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
from config import parser
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
disease_categories = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", 
                      "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", 
                      "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices", "Atelectasis Neg", 
                      "Cardiomegaly Neg", "Consolidation Neg", "Edema Neg", "Enlarged Cardiomediastinum Neg", 
                      "Fracture Neg", "Lung Lesion Neg", "Lung Opacity Neg", "No Finding Neg", "Pleural Effusion Neg", 
                      "Pleural Other Neg", "Pneumonia Neg", "Pneumothorax Neg", "Support Devices Neg"]

anatomy_label_list = ["Right lung", 
                      "Right upper lung zone",
    "Right mid lung zone",
    "Right lower lung zone",
    "Hilar area of right lung",
    "Apical zone of right lung",
    "Right costophrenic sulcus",
    "Right hemidiaphragm",

    "Left lung",
    "Left upper lung zone",
    "Left mid lung zone",
    "Left lower lung zone",
    "Hilar area of left lung",
    "Apical zone of left lung",
    "Left costophrenic sulcus",
    "Left hemidiaphragm",

    "Main Bronchus",
    "Right clavicle",
    "Left clavicle",
    
    "Aortic arch structure",
    "Mediastinum",
    "Superior vena cava structure",
    "cardiac",
    "cavoatrial",
    "Descending aorta",
    "Structure of carina",
]

img_ratio = int(1024/args.img_size)
# 加载 pth 文件
model_dict = torch.load("SavedModels/resnet50_True_MAX_IMG_SIZE_512_num_class_14_epochs_10_best_model.pth")
model_dict = model_dict["model_state_dict"]
model = ResNet50(args).to(device)
model.load_state_dict(model_dict)
model.eval()
target_layers = [model.img_model.layer4[-1]]

finding_anatomy_relation_matrix= np.load("finding_anatomy_relation_matrix.npy")

def graph_construct(img_path, findings_centers_list, anatomy_centers_list, adj_matrix, finding_label, anatomy_label_list):
    img = Image.open(img_path+".jpg").convert("L")
    img = img.resize((1024, 1024))
    img = np.array(img)
    fig = plt.figure(figsize=(10, 10), dpi=120)
    plt.imshow(img, cmap="gray")

    # 绘制 findings 的 nodes
    for i in range(len(findings_centers_list)):
        plt.scatter(findings_centers_list[i][0], findings_centers_list[i][1], s=100, c="xkcd:red")
        plt.text(s=finding_label, x= findings_centers_list[i][0]+10, y= findings_centers_list[i][1]+10, fontdict = dict(fontsize=15, color = "xkcd:orangered"))
    # 绘制 anatomy 的 nodes
    for j in range(len(anatomy_centers_list)):
        plt.scatter(anatomy_centers_list[j][0], anatomy_centers_list[j][1], s=100, c="xkcd:green")
        plt.text(s=anatomy_label_list[j], x=anatomy_centers_list[j][0]+10, y= anatomy_centers_list[j][1]+10, fontdict = dict(fontsize=15, color = "xkcd:sage") )
    # 绘制 edge
    for m in range(len(adj_matrix)):
        for n in range(len(adj_matrix[0])):
            #finding_pos = findings_centers_list[m]
            anatomy_pos = anatomy_centers_list[n]
            #plt.arrow(finding_pos[0], finding_pos[1], anatomy_pos[0]-finding_pos[0], 
                      #anatomy_pos[1]-finding_pos[1], linewidth=adj_matrix[m][n]*4, color="g", length_includes_head=True, )
    #plt.arrow(500, 500, 100, 100, linewidth=4, color="g", length_includes_head=True)
            #plt.annotate(text="", xy=(findings_centers_list[m][0], findings_centers_list[m][1]), 
            #             xytext=(anatomy_centers_list[n][0], anatomy_centers_list[n][1]),
            #             arrowprops=dict(arrowstyle="-", lw=adj_matrix[m][n]*5, color="b", alpha=0),
            #            )
    plt.title("{} graph".format(img_path.split("/")[-1].split(".")[0]))
    plt.axis("off")
    plt.savefig("graph/{}_{}_graph.jpg".format(img_path.split("/")[-1].split(".")[0], finding_label))
    plt.show()

#img_finding_bbox_dict = np.load("img_bbox_dict_test_label_0.npy", allow_pickle=True)
#img_finding_bbox_dict = pickle.load(file = open("img_bbox_test/img_bbox_test_{}.pkl".format(disease_categories[13]), "rb"))
img_anatomy_bbox_dict = np.load("Anatomical_MIMIC/anatomical_mimic_dict.npy", allow_pickle=True)

adj_matrix = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             ])

DATA_DIR = "/drive/mengliang/physionet.org/files/PA_ERECT"
df_test = pd.read_csv(os.path.join(DATA_DIR, "tiny_test.csv"))

img_name_list = ["63b125f7-1b3e7402-94ad99e3-7df00c19-3cf1ea72"]

for i in range(len(img_name_list)):
    img_path = os.path.join(DATA_DIR, "images", img_name_list[i])
    img_original = Image.open(img_path + ".jpg").convert("L")
    img = img_original.resize((args.img_size, args.img_size))

