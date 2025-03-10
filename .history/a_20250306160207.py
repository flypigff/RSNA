import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 路径配置（根据用户截图调整）
TRAIN_DIR = "stage_2_train_images"
TEST_DIR = "stage_2_test_images"
TRAIN_LABELS = "stage_2_train_labels.csv"

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, mode='train'):
        self.dataframe = self._preprocess(dataframe, mode)
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode

    def _preprocess(self, df, mode):
        """处理重复标注和路径问题"""
        df = df[df['Target'] == 1].reset_index(drop=True)
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']
        return df.groupby('patientId').filter(lambda x: len(x) > 0) if mode == 'train' else df

    def __getitem__(self, idx):
        patient_id = self.dataframe.iloc[idx].patientId.strip()  # 处理空格
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        
        # 验证文件存在性
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM文件不存在: {dicom_path}")
            
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1)
        
        if self.mode == 'train':
            boxes = self.dataframe[self.dataframe.patientId == patient_id][
                ['x', 'y', 'x_max', 'y_max']].values.astype(np.float32)
            labels = np.ones(len(boxes))
            
            if self.transforms:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
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

def get_model(num_classes=2, score_thresh=0.5):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = score_thresh
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
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1).transpose(2,0,1)
        image = torch.from_numpy(image).unsqueeze(0).to(device)  # 图像传送到GPU
        
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
                                for s, (x,y,w,h) in zip(scores, boxes)])
        else:
            pred_str = ""
        
        submission.append({'patientId': patient_id, 'PredictionString': pred_str})
    
    return pd.DataFrame(submission)

if __name__ == '__main__':
    # Ensure freeze_support is inside the main block to handle Windows-specific issues
    torch.multiprocessing.freeze_support()

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    train_df = pd.read_csv(TRAIN_LABELS, encoding='utf-8')
    train_dataset = PneumoniaDataset(train_df, TRAIN_DIR, train_transform, 'train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,  # 使用命名函数
        num_workers=2,
        pin_memory=True
    )

    # 初始化模型和优化器
    model = get_model().to(device)  # 将模型移动到GPU
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
