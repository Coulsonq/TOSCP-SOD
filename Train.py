import os
from matplotlib import pyplot as plt
from torch import nn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from Datatset import MyDataset
from tqdm import tqdm
from ViT import VisionTransformer
import torch.nn.functional as F
from torchsummary import summary

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device('cuda')

label_path = os.path.join(r'C:\Users\A\Desktop\ECSSD\GT2')
label_list = os.listdir(label_path)
label_list_path = [os.path.join(label_path, label_name) for label_name in label_list]

cropL_path = r'C:\Users\A\Desktop\ECSSD\cropL'
cropL_list = os.listdir(cropL_path)
cropL_list_path = [os.path.join(cropL_path, cropL_file_name) for cropL_file_name in cropL_list]

feature_path = r'C:\Users\A\Desktop\ECSSD\feature'
feature_list = os.listdir(feature_path)
feature_list_path = [os.path.join(feature_path, feature_name) for feature_name in feature_list]

box_path = r'C:\Users\A\Desktop\ECSSD\box'
box_list = os.listdir(box_path)
box_list_path = [os.path.join(box_path, box_file_name) for box_file_name in box_list]

# 创建数据集
train_dataset = MyDataset(box_list_path, feature_list_path, cropL_list_path, label_list_path)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

model = VisionTransformer()
model.to(device)
pretrained_path = r'D:\object\Semantic_SOD\weight\base_weight.pth'
model.load_state_dict(torch.load(pretrained_path, map_location=device))
model.train()

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

total_params = sum(p.numel() for p in model.parameters())

def train():
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        # 使用 tqdm 包装训练加载器
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for k, (box, feature, label, cropL, idx) in enumerate(train_iterator):
            box = box.to(device)
            feature = feature.to(device)
            label = label.to(device)
            cropL = cropL.to(device)
            score_list = model(feature, box)
            score_list = torch.unsqueeze(score_list, dim=-1)

            weighted_images = cropL * score_list
            #result = torch.sum(weighted_images, dim=1)
            result, _ = torch.max(weighted_images, dim=1)
            clamped_result = torch.clamp(result, 0, 1)

            loss = criterion(clamped_result, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 在 tqdm 的描述中更新损失信息
            train_iterator.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/(k+1):.4f}')

        save_path = os.path.join(r'D:\object\Semantic_SOD\weight', f'ECSSD_epoch_{epoch + 1}_loss_{total_loss}.pth')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path)

        print('Epoch:', epoch, "Total Loss:", total_loss)





if __name__ == '__main__':
    train()