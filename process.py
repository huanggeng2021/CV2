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



train_dir = r"D:\personal-file\Work\CV-HW\data\128\fashion_image_label\train_min"
test_dir = r"D:\personal-file\Work\CV-HW\data\128\fashion_image_label\test_min"
pre_dir = r"D:\personal-file\Work\CV-HW\data\128\exam_fashion"

train_x, train_y = readfile(train_dir, True, False)
print("Size of training data = {}".format(len(train_x)))
#  print(train_x.shape)          128 张 28*28*3的图片
test_x, test_y = readfile(test_dir, True, False)
print("Size of validation data = {}".format(len(test_x)))
pre_x ,pre_y= readfile(pre_dir, True, True)
print("Size of validation data = {}".format(len(pre_x)))


# training 时作 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


batch_size = 8
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(test_x, test_y, test_transform)
pre_set = ImgDataset(pre_x, pre_y, test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
pre_loader = DataLoader(pre_set,batch_size=1,shuffle=False)


for i,data in enumerate(train_loader):
    print(i,data[0].shape)
    break
