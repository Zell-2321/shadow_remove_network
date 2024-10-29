import torch
import time
import psutil
from network import *

device = torch.device("cpu")
batch_size = 1
torch.set_num_threads(4)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

model = UNet().to(device)  # 这里的 ShadowRemovalNet 是你的模型类
model.load_state_dict(torch.load("model/UNet_epoch_50.pth", map_location=device))  # 加载模型权重

dataset = ShadowRemovalDataset(root_dir="test", transform=transform, masked_img=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cpu")
model = model.to(device)

# 开始测试前将模型设为评估模式
model.eval()
total_time = 0
cpu_usages = []

# 不计算梯度以节省内存
with torch.no_grad():
    for data in loader:
        inputs = data[0].to(device)
        
        # 开始时间
        start_time = time.time()
        
        # 获取当前的CPU使用率
        cpu_usage_start = psutil.cpu_percent(interval=None)
        
        # 模型推理
        outputs = model(inputs)
        
        # 结束时间
        end_time = time.time()
        
        # 推理时间
        inference_time = end_time - start_time
        total_time += inference_time
        
        # 推理后的CPU使用率
        cpu_usage_end = psutil.cpu_percent(interval=None)
        
        # 计算平均CPU占用
        average_cpu_usage = (cpu_usage_start + cpu_usage_end) / 2
        cpu_usages.append(average_cpu_usage)

# 计算总推理时间和平均CPU使用率
average_time_per_batch = total_time / len(loader)
average_cpu_usage = sum(cpu_usages) / len(cpu_usages)

print(f"每张图片平均推理时间: {average_time_per_batch:.4f} 秒")
print(f"CPU 平均占用率: {average_cpu_usage:.2f}%")
