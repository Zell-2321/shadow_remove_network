import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 超参数设置
batch_size = 16
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4) 

# 定义数据集类
class ShadowRemovalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(os.path.join(root_dir, root_dir+'_A'))
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        input_image = Image.open(os.path.join(self.root_dir, self.root_dir+'_A', img_name)).convert("RGB")
        mask_image = Image.open(os.path.join(self.root_dir, self.root_dir+'_B', img_name)).convert("RGB")
        target_image = Image.open(os.path.join(self.root_dir, self.root_dir+'_C', img_name)).convert("RGB")
        
        if self.transform:
            input_image = self.transform(input_image)
            mask_image = self.transform(mask_image)
            target_image = self.transform(target_image)
        
        input_data = torch.cat((input_image, mask_image), dim=0)
        
        return input_data, target_image

# 定义U-Net模型（简单示例，可以改为标准U-Net）GAN DIfussion
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 确保输出在[0,1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.decoder(x)
        return x

# 数据增强与加载
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = ShadowRemovalDataset(root_dir="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = UNet().to(device)
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试并可视化
model.eval()
test_dataset = ShadowRemovalDataset(root_dir="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def visualize(input_img, mask_img, output_img, target_img):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    axs[0].imshow(input_img.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Input Image')
    
    axs[1].imshow(mask_img.permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('Mask Image')
    
    axs[2].imshow(output_img.permute(1, 2, 0).cpu().numpy())
    axs[2].set_title('Output Image')
    
    axs[3].imshow(target_img.permute(1, 2, 0).cpu().numpy())
    axs[3].set_title('Target Image')
    
    plt.show()

# 进行测试并显示结果
for inputs, targets in test_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    input_img, mask_img = inputs[:, :3, :, :], inputs[:, 3:, :, :]
    with torch.no_grad():
        outputs = model(inputs)
    
    visualize(input_img[0], mask_img[0], outputs[0], targets[0])
