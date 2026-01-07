import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. 实验配置
# ==========================================
MODEL_NAME = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

# Cityscapes 数据集的 19 个标准类别
# 顺序必须与模型输出类别顺序一致
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 
    'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
]

# ==========================================
# 2. 加载模型与处理器
# ==========================================
print(f"正在加载模型: {MODEL_NAME} ...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()  

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型加载完成，使用设备: {device}")

# ==========================================
# 3. 提取语义特征向量
# ==========================================
def extract_semantic_features(image_path):
    """
    输入: 街景图片路径
    输出: 19维的特征向量
    """
    # A. 图像预处理
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"无法读取图片: {e}")
        return None

    # Hugging Face 的处理器会自动处理 Resize 和 Normalization
    inputs = processor(images=image, return_tensors="pt").to(device)

    # B. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        
    # C. 后处理 logits
    # SegFormer 输出的 logits 尺寸通常是原图的 1/4，需要上采样回原图尺寸
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # 反转为(height, width)
        mode="bilinear",
        align_corners=False,
    )

    # D. 生成语义分割掩膜
    # argmax 获取每个像素概率最大的类别索引 (0-18)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # E. 计算像素占比
    # Pi = (类别 i 的像素数) / (图像总像素数)
    total_pixels = pred_seg.numel()
    pixel_ratios = []
    
    # 遍历 0-18 所有类别索引
    for class_id in range(len(CITYSCAPES_CLASSES)):
        # 计算该类别在掩膜中出现的次数
        class_pixel_count = (pred_seg == class_id).sum().item() #item()实现标量转换
        ratio = class_pixel_count / total_pixels
        pixel_ratios.append(ratio)
        
    return np.array(pixel_ratios)

# ==========================================
# 4. 运行
# ==========================================
if __name__ == "__main__":
    test_image_path = "street_view_sample.png" 
    print(f"正在分析图片: {test_image_path} ...")
    feature_vector = extract_semantic_features(test_image_path)
    
    if feature_vector is not None:
        print("\n=== 实验结果: ===")
        print(f"向量形状: {feature_vector.shape}")
        print("-" * 40)
        # 打印非零的类别，模拟论文中的结果分析
        for cls_name, ratio in zip(CITYSCAPES_CLASSES, feature_vector):
            if ratio > 0.001: # 仅显示占比超过 0.1% 的元素
                print(f"{cls_name.ljust(15)}: {ratio:.4f} ({ratio*100:.2f}%)")
        print("-" * 40)