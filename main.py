


import cv2
import os
import numpy as np
# 对数据进行预处理
import torch
import torchvision
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def readfile(path, label, pre):
    # label 是一个boolean类型的量 ，代表需不需要回传Y值
    image_dir = sorted(os.listdir(path))  # os.listdir 返回指定路径下文件夹和文件列表
    x = np.zeros((len(image_dir), 28, 28, 3), dtype=np.uint8)  # 28*28*3 的图像
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (28, 28))  # 存放进x数组
        if label:
            if pre:
                y[i] = int(file.split(".")[0])
            else:
                y[i] = int(file.split("_")[-1][0])  # 文件名最后一位是标签
    if label:
        return x, y
    else:
        return x










pre_dir = r"D:\personal-file\Work\CV-HW\data\128\exam_fashion"

pre_x ,pre_y= readfile(pre_dir, True, True)
print("Size of validation data = {}".format(len(pre_x)))