import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pydicom
from sklearn.model_selection import train_test_split

# 数据集类
class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.labels.iloc[idx, 0] + '.dcm')
        dicom = pydicom.dcmread(img_path)
        image = dicom.pixel_array
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        bbox = self.labels.iloc[idx, 1:].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, bbox

# 模型定义
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Linear(2048, 4)  # 输出为边界框的4个坐标

    def forward(self, x):
        return self.backbone(x)

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 主函数
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # 加载数据
    print("Loading data...")
    train_labels = pd.read_csv('stage_2_train_labels.csv')
    train_images_dir = 'stage_2_train_images'
    print("Train labels shape:", train_labels.shape)

    # 数据集划分
    print("Splitting dataset...")
    train_labels, val_labels = train_test_split(train_labels, test_size=0.2, random_state=42)
    print("Train dataset size:", len(train_labels))
    print("Validation dataset size:", len(val_labels))

    # 创建数据集
    print("Creating datasets...")
    train_dataset = PneumoniaDataset(train_images_dir, train_labels)
    val_dataset = PneumoniaDataset(train_images_dir, val_labels)

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))

    # 初始化模型、损失函数和优化器
    print("Initializing model...")
    model = PneumoniaModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # 保存模型
    print("Saving model...")
    torch.save(model.state_dict(), 'pneumonia_model.pth')

    # 测试模型
    print("Testing model...")
    model.eval()
    test_images_dir = 'stage_2_test_images'
    test_labels = pd.read_csv('stage_2_sample_submission.csv')

    # 创建测试数据集
    test_dataset = PneumoniaDataset(test_images_dir, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 生成预测结果
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    # 生成提交文件
    submission = pd.DataFrame(predictions, columns=['x', 'y', 'width', 'height'])
    submission['patientId'] = test_labels['patientId']
    submission.to_csv('submission.csv', index=False)

    print("提交文件已生成，可以上传到Kaggle进行评估。")