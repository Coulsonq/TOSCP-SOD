import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import torch.nn.functional as F

def adjust_tensor(tensor_data, target_length=5):
    current_length = tensor_data.shape[0]
    if current_length < target_length:
        # 补零
        padding = torch.ones(target_length - current_length, *tensor_data.shape[1:])
        adjusted_tensor = torch.cat([tensor_data, padding], dim=0)
    elif current_length > target_length:
        # 截断
        adjusted_tensor = tensor_data[:target_length]
    else:
        adjusted_tensor = tensor_data
    return adjusted_tensor

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # 去除每行末尾的换行符并转换为浮点数
    data = [float(line.strip()) for line in data]
    return data

def pad_or_truncate(tensor, target_size, pad_value=0.0):
    current_size = tensor.size(0)
    if current_size < target_size:
        # 需要填充
        padding_size = target_size - current_size
        padding = torch.full((padding_size, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    elif current_size > target_size:
        # 需要截断
        tensor = tensor[:target_size]
    return tensor


def pad_or_truncate_list(tensor_list, desired_length, padding_shape):
    """
    检查 tensor_list 的长度，如果少于 desired_length，则用 padding_shape 的零张量填充；
    如果大于 desired_length，则截断 tensor_list。

    参数:
    tensor_list (list): 包含二维 tensor 的列表。
    desired_length (int): 目标长度。
    padding_shape (tuple): 用于填充的零张量的形状 (height, width)。

    返回:
    list: 处理后的列表。
    """
    current_length = len(tensor_list)
    if current_length < desired_length:
        # 需要填充
        padding_needed = desired_length - current_length
        zero_padding = [torch.zeros(padding_shape) for _ in range(padding_needed)]
        tensor_list.extend(zero_padding)
    elif current_length > desired_length:
        # 需要截断
        tensor_list = tensor_list[:desired_length]

    return tensor_list
'''
# 示例使用
tensor_list = [torch.randn(3, 4) for _ in range(5)]  # 假设这是你的二维 tensor 列表
desired_length = 10
padding_shape = (3, 4)

# 调用函数处理 tensor_list
processed_list = pad_or_truncate_list(tensor_list, desired_length, padding_shape)

# 打印结果
#print(f"Processed list length: {len(processed_list)}")
#print(f"Shape of each tensor: {[tensor.shape for tensor in processed_list]}")
'''

class MyDataset(Dataset):
    def __init__(self, box_path, feature_path, cropL_path, label_path, caption_score_path):
        self.box_path = box_path
        self.feature_path = feature_path
        self.cropL_path = cropL_path
        self.label_path = label_path
        self.transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor()
        ])
        self.to_tensor = transforms.ToTensor()
        self.caption_score_path = caption_score_path

    def __len__(self):
        return len(self.label_path)

    def __getitem__(self, idx):
        box_path = self.box_path[idx]
        feature_path = self.feature_path[idx]
        cropL_path = self.cropL_path[idx]
        label_path = self.label_path[idx]
        caption_score_path = self.caption_score_path[idx]

        # 读取图像和标签
        label = Image.open(label_path).convert('L')  # 假设label为灰度图

        label = self.transform(label)
        label = torch.squeeze(label)

        # 读取vec_dir下所有的txt文件
        feature_files = glob.glob(os.path.join(feature_path, '*.txt'))
        features = []
        for vf in feature_files:
            with open(vf, 'r') as file:
                # 读取文件内容并将逗号替换为空格
                content = file.read().strip().replace(',', ' ')
                # 将内容转换为浮点数列表
                feature = np.array([float(num) for num in content.split()])
                features.append(torch.tensor(feature, dtype=torch.float32))
        # 堆叠向量
        stacked_features = torch.stack(features)
        padded_features = pad_or_truncate(stacked_features, 5, 0)

        box_list = os.listdir(box_path)
        boxes = []
        for box_name in box_list:
            box_loc = os.path.join(box_path, box_name)
            with open(box_loc, 'r') as file:
                # 读取文件内容
                data = file.read().strip()
                # 将字符串转换为浮点数列表
                float_list = [float(num) for num in data.split(',')]
                # 将浮点数列表转换为 PyTorch 的 tensor
                boxes.append(torch.tensor(float_list, dtype=torch.float32))
        stacked_boxes = torch.stack(boxes)
        padded_boxes = pad_or_truncate(stacked_boxes, 5, 0)

        cropL_img_list = []
        for cropL_name in os.listdir(cropL_path):
            cropL_one_path = os.path.join(cropL_path, cropL_name)
            cropL = Image.open(cropL_one_path).convert('L')
            #cropL = self.to_tensor(cropL)
            cropL = self.transform(cropL)
            cropL = torch.squeeze(cropL)
            cropL_img_list.append(cropL)
        padded_cropL_img_list = pad_or_truncate_list(cropL_img_list, 5, (cropL.shape[0], cropL.shape[1]))
        stacked_cropL_tensor = torch.stack(padded_cropL_img_list)

        all_score = []
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(caption_score_path):
            file_path = os.path.join(caption_score_path, file_name)
            file_data = read_txt_file(file_path)
            all_score.extend(file_data)
        # 将所有数据合并成一个张量
        tensor_data = torch.tensor(all_score)

        return padded_boxes, padded_features, label, stacked_cropL_tensor, f"{idx+1:06d}", tensor_data


if __name__ == '__main__':

    label_path = os.path.join(r'C:\Users\A\Desktop\2d_dataset2\labels')
    label_list = os.listdir(label_path)
    label_list_path = [os.path.join(label_path, label_name) for label_name in label_list]

    cropL_path = r'C:\Users\A\Desktop\cropL'
    cropL_list = os.listdir(cropL_path)
    cropL_list_path = [os.path.join(cropL_path, cropL_file_name) for cropL_file_name in cropL_list]

    feature_path = r'C:\Users\A\Desktop\feature'
    feature_list = os.listdir(feature_path)
    feature_list_path = [os.path.join(feature_path, feature_name) for feature_name in feature_list]

    box_path = r'C:\Users\A\Desktop\box'
    box_list = os.listdir(box_path)
    box_list_path = [os.path.join(box_path, box_file_name) for box_file_name in box_list]

    # 创建数据集
    dataset = MyDataset(box_list_path, feature_list_path, cropL_list_path, label_list_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # 遍历数据
    for box, feature, label, cropL, idx in dataloader:
        print(stacked_boxes.shape, stacked_features.shape, len(cropL_img_list), label.shape, idx)
