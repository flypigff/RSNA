import os
import pydicom
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 配置文件路径
TRAIN_IMG_DIR = "./stage_2_train_images"
TEST_IMG_DIR = "./stage_2_test_images"
TRAIN_LABELS_PATH = "./stage_2_train_labels.csv"
CLASS_INFO_PATH = "./stage_2_detailed_class_info.csv"

# 数据预处理类
class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.dataframe = self._preprocess_data(dataframe)
        self.image_dir = image_dir
        self.transforms = transforms

    def _preprocess_data(self, df):
        """处理重复的边界框标注"""
        df = df[df['Target'] == 1].reset_index(drop=True)
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']
        return df

    def __getitem__(self, idx):
        patient_id = self.dataframe.iloc[idx]['patientId']
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        
        # DICOM窗宽窗位处理
        intercept = dicom.RescaleIntercept if 'RescaleIntercept' in dicom else 0.0
        slope = dicom.RescaleSlope if 'RescaleSlope' in dicom else 1.0
        image = slope * dicom.pixel_array.astype(np.float32) + intercept
        
        # 归一化并转换为三通道
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1)
        
        # 获取当前患者的边界框
        boxes = self.dataframe[self.dataframe['patientId'] == patient_id][['x', 'y', 'x_max', 'y_max']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # 数据增强
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        transformed = self.transforms(
            image=image,
            bboxes=boxes.numpy().tolist(),  # 转换为列表格式
            labels=np.ones(len(boxes))      # 显式传递标签
        )
        boxes = torch.as_tensor(transformed['bboxes'])
        labels = torch.as_tensor(transformed['labels'])
        return image, target

    def __len__(self):
        return len(self.dataframe['patientId'].unique())

# 数据增强配置
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc'))

# 初始化数据加载器
train_df = pd.read_csv(TRAIN_LABELS_PATH)
train_dataset = PneumoniaDataset(train_df, TRAIN_IMG_DIR, train_transform)
train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x)),
    num_workers=0
)

# 模型配置
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

# 预测与提交生成
def predict_submission(model, test_dir):
    model.eval()
    submission = []
    
    for filename in os.listdir(test_dir):
        if not filename.endswith('.dcm'):
            continue
            
        patient_id = filename[:-4]
        dicom_path = os.path.join(test_dir, filename)
        dicom = pydicom.dcmread(dicom_path)
        
        # 预处理
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=0)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(image)
        
        # 后处理
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # 应用NMS
        keep = torchvision.ops.nms(
            torch.as_tensor(boxes),
            torch.as_tensor(scores),
            iou_threshold=0.2
        )
        
        pred_str = ' '.join([f"{scores[i]:.4f} {boxes[i][0]:.1f} {boxes[i][1]:.1f} {boxes[i][2]:.1f} {boxes[i][3]:.1f}" 
                            for i in keep])
        submission.append({'patientId': patient_id, 'PredictionString': pred_str})
    
    return pd.DataFrame(submission)

submission_df = predict_submission(model, TEST_IMG_DIR)
submission_df.to_csv("submission.csv", index=False)