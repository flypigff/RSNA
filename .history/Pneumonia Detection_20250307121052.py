import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 路径配置（根据用户截图调整）
TRAIN_DIR = "stage_2_train_images"
TEST_DIR = "stage_2_test_images"
TRAIN_LABELS = "stage_2_train_labels.csv"
DETAILED_CLASS_INFO = "stage_2_detailed_class_info.csv"

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, mode='train', class_info=None):
        """
        初始化数据集类
        :param dataframe: 数据集的DataFrame，包含标注信息
        :param image_dir: 图像所在目录
        :param transforms: 数据增强函数
        :param mode: 'train' 或 'test'
        :param class_info: 类别信息 (从stage_2_detailed_class_info.csv读取)
        """
        self.class_info = class_info  # 将class_info赋值给类的实例属性
        self.dataframe = self._preprocess(dataframe, mode)
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode

    def _preprocess(self, df, mode):
        """处理重复标注和路径问题"""
        # 读取并添加类别信息
        df = df.merge(self.class_info[['patientId', 'class']], on='patientId', how='left')
        df = df[df['class'] != 'No Lung Opacity / Not Normal'].reset_index(drop=True)  # 排除 "Not Normal" 类别
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']
        
        # 清理含有 NaN 值的边界框
        df = df.dropna(subset=['x', 'y', 'x_max', 'y_max'])  # 丢弃包含 NaN 的样本
        
        # 过滤掉无效的边界框（边界框坐标为负数）
        df = df[df['x'] >= 0]
        df = df[df['y'] >= 0]
        df = df[df['x_max'] >= 0]
        df = df[df['y_max'] >= 0]
        
        return df.groupby('patientId').filter(lambda x: len(x) > 0) if mode == 'train' else df

    def __getitem__(self, idx):
        patient_id = self.dataframe.iloc[idx].patientId.strip()  # 处理空格
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        # 验证文件是否存在
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM文件不存在: {dicom_path}")
            
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())  # 归一化
        image = np.stack([image]*3, axis=-1)
        
        if self.mode == 'train':
            boxes = self.dataframe[self.dataframe.patientId == patient_id][
                ['x', 'y', 'x_max', 'y_max']].values.astype(np.float32)
            labels = self.dataframe[self.dataframe.patientId == patient_id]['class'].values  # 获取类别信息
            
            # 将类别信息映射为标签
            label_map = {"No Lung Opacity / Not Normal": 0, "Normal": 1, "Lung Opacity": 2}
            labels = np.array([label_map[label] for label in labels], dtype=np.int64)  # 将类别转换为数字
            
            # 检查边界框是否有效，避免 NaN 值
            if np.any(np.isnan(boxes)):  # 如果存在 NaN 值，跳过这个样本
                return None  # 返回 None 代表跳过该样本
            
            if self.transforms:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed['image']
                boxes = np.array(transformed['bboxes'])
            
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
            return image, target
        
        return image, patient_id

    def __len__(self):
        return len(self.dataframe.patientId.unique())



# 数据增强修正
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(
        size=(512, 512),
        scale=(0.08, 1.0),
        ratio=(0.75, 1.33)
    ),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
    min_visibility=0.3
))

def get_model(num_classes=3, score_thresh=0.5):
    # 使用ResNet作为Faster R-CNN的backbone
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)  # 或者可以替换为 'resnet101'
    model = FasterRCNN(backbone, num_classes=num_classes)
    model.roi_heads.score_thresh = score_thresh  # 设置评分阈值
    return model

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc='Training')
    
    for images, targets in progress:
        images = [img.to(device) for img in images]  # 将图像传送到GPU
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 将目标传送到GPU
        
        # 使用模型进行前向传播
        outputs = model(images, targets)
        
        losses = sum(loss for loss in outputs.values())  # 计算所有损失
        
        optimizer.zero_grad()  # 清除上一轮的梯度
        losses.backward()  # 反向传播计算梯度
        optimizer.step()  # 优化器步进，更新参数
        
        total_loss += losses.item()  # 累计损失
        progress.set_postfix(loss=total_loss / (progress.n + 1))  # 打印损失
    
    return total_loss / len(loader)

def collate_fn(batch):
    images = [item[0] for item in batch]  # images列表
    targets = [item[1] for item in batch]  # targets列表
    return images, targets

def predict(model, test_dir, device, threshold=0.3):
    model.eval()
    submission = []
    test_files = [f.replace(' ', '') for f in os.listdir(test_dir) if f.endswith('.dcm')]  # 处理空格
    
    progress = tqdm(test_files, desc='Predicting')
    for filename in progress:
        patient_id = filename[:-4]
        dicom_path = os.path.join(test_dir, filename)
        
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())  # 归一化
        image = np.stack([image] * 3, axis=-1).transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).to(device)  # 将图像传送到GPU
        
        with torch.no_grad():
            preds = model(image)
        
        boxes = preds[0]['boxes'].cpu().numpy()
        scores = preds[0]['scores'].cpu().numpy()
        
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        if len(scores) > 0:
            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            pred_str = ' '.join([f"{s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" 
                                for s, (x, y, w, h) in zip(scores, boxes)])
        else:
            pred_str = ""
        
        submission.append({'patientId': patient_id, 'PredictionString': pred_str})
    
    return pd.DataFrame(submission)

if __name__ == '__main__':
    # 确保Windows系统中多进程的兼容性
    torch.multiprocessing.freeze_support()

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载类别信息
    class_info_df = pd.read_csv(DETAILED_CLASS_INFO)
    
    # 加载训练数据
    train_df = pd.read_csv(TRAIN_LABELS, encoding='utf-8')
    train_dataset = PneumoniaDataset(train_df, TRAIN_DIR, train_transform, 'train', class_info=class_info_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,  # 使用命名函数
        num_workers=2,
        pin_memory=True
    )

    # 初始化模型和优化器
    model = get_model().to(device)  # 将模型移到GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 训练循环
    print("Starting training...")
    for epoch in range(10):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    # 生成提交文件
    submission_df = predict(model, TEST_DIR, device, threshold=0.3)
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved!")
