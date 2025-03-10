# 肺炎检测完整代码（修正版）
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

# 配置路径
TRAIN_IMG_DIR = "./stage_2_train_images"
TEST_IMG_DIR = "./stage_2_test_images"
TRAIN_LABELS_PATH = "./stage_2_train_labels.csv"

# 数据增强配置（关键修正点 [[1]](#__1)）
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],  # 关键修正点1
        min_visibility=0.3
    )
)

class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.dataframe = self._preprocess(dataframe)
        self.image_dir = image_dir
        self.transforms = transforms

    def _preprocess(self, df):
        """处理重复标注（[[3]](#__3)）"""
        df = df[df['Target'] == 1].reset_index(drop=True)
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']
        return df.groupby('patientId').filter(lambda x: len(x) > 0)

    def __getitem__(self, idx):
        patient_id = self.dataframe.iloc[idx]['patientId']
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        
        # DICOM预处理（[[0]](#__0)）
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1)
        
        # 获取对应边界框（关键修正点2 [[1]](#__1)）
        boxes = self.dataframe[self.dataframe['patientId'] == patient_id][
            ['x', 'y', 'x_max', 'y_max']].values.tolist()
        labels = np.ones(len(boxes))  # 所有框标记为肺炎阳性

        # 应用数据增强
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels  # 显式传递标签
            )
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros_like(labels)
        }
        return image, target

    def __len__(self):
        return len(self.dataframe['patientId'].unique())

def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    return model

if __name__ == '__main__':  # 关键修正点3（解决多进程错误）
    import multiprocessing
    multiprocessing.freeze_support()

    # 初始化配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 数据加载器配置（关键修正点4）
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    train_dataset = PneumoniaDataset(train_df, TRAIN_IMG_DIR, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0  # Windows下必须设为0
    )

    # 训练循环
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "pneumonia_model.pth")
