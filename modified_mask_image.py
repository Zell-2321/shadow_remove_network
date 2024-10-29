import cv2
import numpy as np

# 读取图像
image = cv2.imread('fig5.png')

# 检查图像是否正确读取
if image is None:
    raise ValueError("图像读取失败，请检查文件路径")

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 设置阈值，纯白色区域(255)变为黑色，其他区域变为白色
_, thresholded = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)

# 保存输出结果
cv2.imwrite('fig5_2.png', thresholded)

# 显示原图和处理后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
