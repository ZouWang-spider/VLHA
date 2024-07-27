import torch
from transformers import ViTModel, ViTConfig
from PIL import Image
import numpy as np
import requests
from torchvision import transforms

#process image
def preprocess_image(image_path, image_size=224):
    # 打开图像并转换为 RGB 模式
    image = Image.open(image_path).convert('RGB')
    # 定义图像预处理变换
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # 应用预处理变换
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# # 图像文件路径
# image_path = "E:/PythonProject2/VLHA/Datasets/227313.jpg"
#
# # 预处理图像
# input_tensor = preprocess_image(image_path)
# model = ViTModel.from_pretrained('E:/vit-base-patch16-224-in21k')
#
# # 获取图片的表征
# with torch.no_grad():
#     outputs = model(pixel_values=input_tensor)
#
# # 提取图片的特征向量
# # outputs.last_hidden_state 的形状是 [batch_size, num_patches, hidden_size]
# # 通常使用第一个 patch 的特征作为图片的全局表征
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)   #torch.Size([1, 197, 768]) ,[CLS]+196patches
#
# # 使用 squeeze() 函数压缩维度
# output_tensor = last_hidden_states.squeeze(0)
# print(output_tensor.shape)


