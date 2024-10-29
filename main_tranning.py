# from network import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from network import *


import time


torch.set_num_threads(4)

def main() -> None:
    # Step1:
    # 训练一个一个网络，用于生成Mask，并保存在network目录下的模型文件中。网络名称_epochNum_version.pth

    ## 超参数设置
    batch_size = 16
    num_epochs = 50 #50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = ShadowRemovalDataset(root_dir="train", transform=transform, masked_img=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = UNet_mask().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Step2:
    # 将二值化后的Mask以及原图作为新网络的输入，生成去除阴影后的图像
    train_dataset = ShadowRemovalDataset(root_dir="test", transform=transform, masked_img=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    time.sleep(10)


if __name__ == '__main__':
    main()