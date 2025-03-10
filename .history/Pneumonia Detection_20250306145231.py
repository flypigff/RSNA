import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# 配置文件路径
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
        """处理重复标注和阴性样本"""
        df = df[df.Target == 1].reset_index(drop=True)
        df['x_max'] = df['x'] + df['width']
        df['y_max'] = df['y'] + df['height']
        return df.groupby('patientId').filter(lambda x: len(x) > 0) if mode == 'train' else df

    def __getitem__(self, idx):
        patient_id = self.dataframe.iloc[idx].patientId
        dicom_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        
        # DICOM预处理
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1)
        
        # 训练模式处理
        if self.mode == 'train':
            boxes = self.dataframe[self.dataframe.patientId == patient_id][
                ['x', 'y', 'x_max', 'y_max']].values
            labels = np.ones(len(boxes))
            
            if self.transforms:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['labels']
            
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
            return image, target
        
        return image, patient_id

    def __len__(self):
        return len(self.dataframe.patientId.unique())

# 数据增强配置
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0)),  # 修正后的参数
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
    min_visibility=0.3
))

def get_model(num_classes=2, score_thresh=0.5):
    """初始化Faster R-CNN模型"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.score_thresh = score_thresh
    return model

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc='Training')

    # 在训练循环前添加数据验证（[[4]](#__4)）
    for batch_idx, (images, targets) in enumerate(train_loader):
        assert isinstance(images, list), "Images应为列表形式"
        assert isinstance(targets, list), "Targets应为字典列表"
        break  # 验证一个批次后退出

    
    for images, targets in progress:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 正确接收模型输出（[[5]](#__5)）
        outputs = model(images, targets)
        losses = sum(loss for loss in outputs.values())  # 单变量接收字典

        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        progress.set_postfix(loss=total_loss/(progress.n+1))
    
    return total_loss / len(loader)

def predict(model, test_dir, device, threshold=0.3):
    """生成符合Kaggle要求的预测结果"""
    model.eval()
    submission = []
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.dcm')]
    
    progress = tqdm(test_files, desc='Predicting')
    for filename in progress:
        patient_id = filename[:-4]
        dicom_path = os.path.join(test_dir, filename)
        dicom = pydicom.dcmread(dicom_path)
        
        # 预处理
        image = dicom.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        image = np.stack([image]*3, axis=-1).transpose(2,0,1)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            preds = model(image)
        
        # 后处理
        boxes = preds[0]['boxes'].cpu().numpy()
        scores = preds[0]['scores'].cpu().numpy()
        
        # 应用阈值和非极大抑制
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        if len(scores) > 0:
            # 按置信度降序排列
            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            
            # 转换格式为(x, y, width, height)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
            
            pred_str = ' '.join([f"{s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" 
                                for s, (x,y,w,h) in zip(scores, boxes)])
        else:
            pred_str = ""
        
        submission.append({'patientId': patient_id, 'PredictionString': pred_str})
    
    return pd.DataFrame(submission)

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    
    # 初始化配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据加载
    train_df = pd.read_csv(TRAIN_LABELS)
    train_dataset = PneumoniaDataset(train_df, TRAIN_DIR, train_transform, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )
    
    # 模型训练
    model = get_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("Starting training...")
    for epoch in range(10):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    
    # 生成提交文件
    submission_df = predict(model, TEST_DIR, device, threshold=0.3)
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file saved!")