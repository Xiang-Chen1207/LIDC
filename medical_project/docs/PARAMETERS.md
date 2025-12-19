# LIDC-IDRI U-Net 消融实验配置与指标说明

## 目录
- [评估指标](#评估指标)
- [模型参数](#模型参数)
- [预处理方法参数](#预处理方法参数)
- [训练配置](#训练配置)

---

## 评估指标

### 1. Dice 系数 (Dice Coefficient / F1-Score for Segmentation)

Dice 系数衡量预测区域与真实区域的重叠程度，是医学图像分割中最常用的指标。

**公式:**
$$
\text{Dice} = \frac{2 \times |A \cap B|}{|A| + |B|} = \frac{2 \times TP}{2 \times TP + FP + FN}
$$

**代码实现:**
```python
intersection = (pred * target).sum()
union = pred.sum() + target.sum()
dice = (2.0 * intersection + smooth) / (union + smooth)
```

**取值范围:** [0, 1]，越接近 1 表示分割效果越好

---

### 2. IoU (Intersection over Union / Jaccard Index)

IoU 计算预测区域与真实区域的交集与并集之比。

**公式:**
$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}
$$

**代码实现:**
```python
intersection = (pred * target).sum()
union = pred.sum() + target.sum() - intersection
iou = (intersection + smooth) / (union + smooth)
```

**取值范围:** [0, 1]，越接近 1 表示分割效果越好

**Dice 与 IoU 的关系:**
$$
\text{Dice} = \frac{2 \times \text{IoU}}{1 + \text{IoU}}
$$

---

### 3. Precision (精确率)

精确率衡量预测为正类的样本中，有多少是真正的正类。

**公式:**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**代码实现:**
```python
tp = (pred * target).sum()
fp = (pred * (1 - target)).sum()
precision = (tp + smooth) / (tp + fp + smooth)
```

**含义:** 预测的结节区域中，有多少比例是真正的结节

---

### 4. Recall (召回率 / Sensitivity / 敏感度)

召回率衡量真正的正类样本中，有多少被正确预测。

**公式:**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**代码实现:**
```python
tp = (pred * target).sum()
fn = ((1 - pred) * target).sum()
recall = (tp + smooth) / (tp + fn + smooth)
```

**含义:** 真正的结节区域中，有多少比例被检测出来

---

### 5. F1 Score

F1 分数是 Precision 和 Recall 的调和平均数。

**公式:**
$$
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**代码实现:**
```python
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
```

**注意:** 对于二分割任务，F1 Score 等价于 Dice 系数

---

### 6. Specificity (特异度)

特异度衡量真正的负类样本中，有多少被正确预测为负类。

**公式:**
$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

**含义:** 非结节区域中，有多少比例被正确识别为非结节

---

### 混淆矩阵说明

| | 预测为正 (Positive) | 预测为负 (Negative) |
|---|---|---|
| **实际为正** | TP (真阳性) | FN (假阴性) |
| **实际为负** | FP (假阳性) | TN (真阴性) |

- **TP (True Positive):** 正确预测的结节像素
- **TN (True Negative):** 正确预测的非结节像素
- **FP (False Positive):** 错误预测为结节的像素
- **FN (False Negative):** 漏检的结节像素

---

## 模型参数

### U-Net 架构

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 128 × 128 × 3 | RGB 三通道图像 |
| 输出尺寸 | 128 × 128 × 1 | 二值分割掩码 |
| 编码器滤波器 | [64, 128, 256, 512] | 逐层加倍 |
| 桥接层滤波器 | 1024 | 最深层 |
| Dropout 率 | 0.3 | 防止过拟合 |

### 模型类型

| 类型 | 编码器滤波器 | 参数量 | 适用场景 |
|------|-------------|-------|---------|
| small | [32, 64, 128, 256] | ~2M | 快速实验 |
| standard | [64, 128, 256, 512] | ~8M | 默认选择 |
| large | [64, 128, 256, 512, 1024] | ~31M | 大规模数据 |

---

## 预处理方法参数

### 1. CLAHE (对比度受限自适应直方图均衡)

| 参数 | 值 | 说明 |
|------|-----|------|
| clipLimit | 2.0 | 对比度限制阈值 |
| tileGridSize | (8, 8) | 分块大小 |

**作用:** 增强图像对比度，使结节边界更清晰

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(image)
```

---

### 2. 中值滤波 (Median Filter)

| 参数 | 值 | 说明 |
|------|-----|------|
| kernel_size | 3 | 滤波核大小 |

**作用:** 去除椒盐噪声，保持边缘

```python
denoised = cv2.medianBlur(image, 3)
```

---

### 3. 高斯平滑 (Gaussian Blur)

| 参数 | 值 | 说明 |
|------|-----|------|
| kernel_size | (3, 3) | 滤波核大小 |
| sigma | 0 | 标准差 (0 表示自动计算) |

**作用:** 平滑图像，去除高频噪声

```python
denoised = cv2.GaussianBlur(image, (3, 3), 0)
```

---

### 4. 边缘检测 (Canny Edge Detection + Morphology)

| 参数 | 值 | 说明 |
|------|-----|------|
| low_threshold | 50 | Canny 低阈值 |
| high_threshold | 150 | Canny 高阈值 |
| morph_kernel | (5, 5) | 形态学闭运算核大小 |
| morph_iterations | 2 | 闭运算迭代次数 |
| min_contour_area | 200 | 最小轮廓面积 |

**作用:** 提取 ROI 区域掩码

```python
edges = cv2.Canny(image, 50, 150)
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
```

---

### 5. GHPF (高斯高通滤波器)

| 参数 | 值 | 说明 |
|------|-----|------|
| kernel_size | (15, 15) | 高斯核大小 |
| sigma | 3 | 高斯标准差 |
| offset | 127 | 避免负值的偏移 |

**作用:** 增强边缘和细节，突出结节特征

```python
lowpass = cv2.GaussianBlur(image, (15, 15), 3)
highpass = cv2.subtract(image, lowpass)
highpass = cv2.add(highpass, 127)
```

---

## 预处理方法组合

| 方法ID | 方法名称 | 流程 |
|--------|---------|------|
| - | clahe | CLAHE |
| - | clahe_median | CLAHE → 中值滤波 |
| - | clahe_smooth | CLAHE → 高斯平滑 |
| C3 | clahe_median_edge_ghpf | CLAHE → 中值滤波 → 边缘检测 → GHPF |
| C7 | clahe_smooth_edge_ghpf | CLAHE → 高斯平滑 → 边缘检测 → GHPF |

---

## 训练配置

### 基本参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 32 | 批次大小 |
| epochs | 100 | 训练轮数 |
| learning_rate | 1e-4 | 初始学习率 |
| weight_decay | 1e-5 | 权重衰减 |
| optimizer | AdamW | 优化器 |
| loss_function | BCE + Dice | 损失函数 |

### 学习率调度

| 参数 | 值 | 说明 |
|------|-----|------|
| scheduler | ReduceLROnPlateau | 调度器类型 |
| mode | max | 监控指标模式 (Dice) |
| factor | 0.5 | 衰减因子 |
| patience | 5 | 等待轮数 |
| min_lr | 1e-7 | 最小学习率 |

### 早停策略

| 参数 | 值 | 说明 |
|------|-----|------|
| patience | 15 | 等待轮数 |
| monitor | val_dice | 监控指标 |

### 数据增强

| 增强方式 | 参数 |
|---------|------|
| 水平翻转 | 50% 概率 |
| 垂直翻转 | 50% 概率 |
| 随机旋转 | ±15° |
| 亮度调整 | ±10% |
| 对比度调整 | ±10% |

---

## 数据集划分

| 数据集 | 比例 |
|-------|-----|
| 训练集 | 70% |
| 验证集 | 20% |
| 测试集 | 10% |

---

## 使用示例

### 训练

```bash
# 使用 CLAHE + 中值 + 边缘检测 + GHPF 方法 (C3)
python train.py --preprocess clahe_median_edge_ghpf --epochs 100 --gpu 0

# 使用 CLAHE + 平滑 + 边缘检测 + GHPF 方法 (C7)
python train.py --preprocess clahe_smooth_edge_ghpf --epochs 100 --gpu 0
```

### 评估

```bash
python evaluate.py /path/to/best_model.pth --preprocess clahe_median_edge_ghpf
```

---

## 参考文献

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
3. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
4. Salehi, S. S. M., et al. (2017). Tversky Loss Function for Image Segmentation.
