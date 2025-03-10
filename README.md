# RSNA肺炎检测挑战赛解决方案 


医学影像目标检测项目，基于Faster R-CNN实现肺炎病灶检测，兼容Kaggle竞赛提交格式。

##  仓库结构
```bash
.
├── Pneumonia Detection.py      # 主程序（训练+推理）
├── README.md                   # 项目文档
├── config/                     # 配置文件
│   └── paths.py                # 路径配置
├── data/                       # 数据集（.gitignore）
│   ├── stage_2_test_images/    
│   ├── stage_2_train_images/
│   ├── stage_2_train_labels.csv
│   └── stage_2_detailed_class_info.csv
├── submissions/               # 生成结果
│   ├── submission1.csv         # 初始结果
│   └── submission2.csv         # 优化后结果
└── requirements.txt           # 依赖库
```

##  快速使用
```bash
# 克隆仓库
git clone https://github.com/flypigff/RSNA.git
cd RSNA

# 安装依赖
pip install -r requirements.txt

# 训练模型 (需先下载数据集到data目录)
python "Pneumonia Detection.py" --epochs 20 --batch_size 4

# 生成提交文件
python "Pneumonia Detection.py" --mode predict --threshold 0.3
```

##  模型架构
```python
Faster R-CNN with:
├── Backbone: ResNet-101 + FPN 
├── ROI Heads: 
│   ├── Box Head: 2 FC Layers
│   └── Score Threshold: 0.5
└── Optimizer: AdamW(lr=1e-4, weight_decay=1e-4)
```

##  数据流程
### 预处理流程
```python
1. DICOM → 归一化 → RGB转换
2. 异常值过滤：
   - 去除'No Lung Opacity / Not Normal'类别
   - 删除含NaN的边界框
   - 过滤负坐标值
3. 数据增强：
   ├─ HorizontalFlip(0.5)
   ├─ RandomResizedCrop(512x512)
   └─ ShiftScaleRotate(0.1/0.2/15°)
```

##  提交结果对比
| 版本 | private score  | public score  |
|------|------|----------------------|
| v1   | 0.05054  | 0.02149                |
| v2   | 0.07339  | 0.02986              |

##  注意事项
1. 数据集需从Kaggle手动下载后放入`data/`目录
2. 训练时自动跳过无效样本（返回None的情况）
3. Windows用户需添加：
   ```python
   if __name__ == '__main__':
       torch.multiprocessing.freeze_support()
   ```
4. 推荐使用CUDA 11.3+环境
