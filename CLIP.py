import os
import clip
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from CLIP_dataset import ImageTextDataset
import torch.nn.functional as F
import numpy as np

# 加载预训练的CLIP模型
device = torch.device('cuda')
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()
criterion = torch.nn.BCELoss().to(device)
optimizer = Adam(model.parameters(), lr=0.0001)

texts_main_path = r'C:\Users\A\Desktop\duts2\caption'
text_list = os.listdir(texts_main_path)
text_list_path = [os.path.join(texts_main_path, text_name) for text_name in text_list]

crops_main_path = r'C:\Users\A\Desktop\duts2\crop'
crop_dir_list = os.listdir(crops_main_path)
crop_list_path = [os.path.join(crops_main_path, crop_name) for crop_name in crop_dir_list]

label_path = os.path.join(r'C:\Users\A\Desktop\duts2\GT')
label_list = os.listdir(label_path)
label_list_path = [os.path.join(label_path, label_name) for label_name in label_list]

cropL_path = r'C:\Users\A\Desktop\duts2\cropL'
cropL_list = os.listdir(cropL_path)
cropL_list_path = [os.path.join(cropL_path, cropL_file_name) for cropL_file_name in cropL_list]

dataset = ImageTextDataset(crop_list_path, text_list_path, preprocess, cropL_list_path, label_list_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def repeat_tensor(tensor, target_shape):
    """
    将输入张量以相同内容复制到指定的目标形状。

    参数:
    tensor (torch.Tensor): 需要复制的张量，形状为 (1, 768)。
    target_shape (tuple): 目标形状，例如 (5, 768)。

    返回:
    torch.Tensor: 复制后的张量。
    """
    current_shape = tensor.shape
    if current_shape[1] != target_shape[1]:
        raise ValueError("目标形状的列数必须与输入张量的列数相同")

    repeats = target_shape[0] // current_shape[0]
    repeated_tensor = tensor.repeat(repeats, 1)

    return repeated_tensor

def pad_or_truncate_tensor(tensor, target_shape):
    """
    将输入张量补零或截断到指定的目标形状。

    参数:
    tensor (torch.Tensor): 需要处理的张量。
    target_shape (tuple): 目标形状。

    返回:
    torch.Tensor: 补零或截断后的张量。
    """
    current_shape = tensor.shape

    if len(current_shape) != len(target_shape):
        raise ValueError("目标形状的维度必须与输入张量的维度相同")

    # 截断张量到目标形状
    slices = tuple(slice(0, min(current, target)) for current, target in zip(current_shape, target_shape))
    tensor = tensor[slices]

    # 计算需要补零的数量
    pad_sizes = []
    for i in range(len(current_shape) - 1, -1, -1):
        pad_size = target_shape[i] - tensor.shape[i]
        pad_sizes.extend([0, pad_size])

    # 使用 pad 函数补零
    padded_tensor = F.pad(tensor, pad_sizes, mode='constant', value=0)
    return padded_tensor

save_main_path = r'C:\Users\A\Desktop\duts2\clip_score'

for crops, text, cropL, label, idx in dataloader:
    crops = crops.to(device)
    text = text.to(device)
    cropL = cropL.to(device)
    label = label.to(device)
    print(idx.item())
    # 获取图像和文本的特征向量
    image_features = model.encode_image(crops.squeeze(dim=0))
    text_features = model.encode_text(text)

    # 计算相似度分数
    logits_per_image = image_features @ text_features.T
    logits_per_image = logits_per_image.squeeze()

    logits_per_image /= 100.0
    score = logits_per_image.cpu().detach().numpy()

    save_dir = os.path.join(save_main_path, f'{idx.item():06}.txt')

    if score.ndim == 0:
        score = score.reshape(1)

    with open(save_dir, 'w') as f:
        for value in score:
            f.write(f"{value}\n")

    print(f"Tensor values have been saved to {save_dir}")



