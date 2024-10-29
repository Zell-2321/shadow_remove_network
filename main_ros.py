from network import *


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

if __name__ == '__main__':

    image_name = "fig3.png"
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    device = torch.device("cpu")

    model = UNet_mask().to(device)  # 这里的 ShadowSegmentationNet 是你的模型类
    model.load_state_dict(torch.load("model/UNet_mask_epoch_50.pth", map_location=device))  # 加载模型权重

    # # 测试并可视化
    model.eval()

    input_img = Image.open(image_name).convert("RGB")
    input_img = transform(input_img)
    input_img = input_img.unsqueeze(0).to(device)


    with torch.no_grad():
        output_img = model(input_img)

    output_img = output_img.squeeze(0)  # 去掉批次维度
    output_img = (output_img > 0.5).float()

    input_img = Image.open(image_name).convert("RGB")
    input_img = transform(input_img)
    output_img = Image.open('fig5_2.png').convert("RGB")
    output_img = transform(output_img)
    combined_img = torch.cat((input_img, output_img), dim=0)
    print(combined_img.shape)
    combined_img = combined_img.unsqueeze(0).to(device)

    model = UNet().to(device)  # 这里的 ShadowRemovalNet 是你的模型类
    model.load_state_dict(torch.load("model/UNet_epoch_50.pth", map_location=device))  # 加载模型权重

    # # 测试并可视化
    model.eval()

    with torch.no_grad():
        output_img_2 = model(combined_img)

    print("output_img.shape:", output_img.shape)
    visualize(input_img, output_img, output_img_2[0], input_img)
    
    

    
    
