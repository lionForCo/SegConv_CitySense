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
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 
    'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
]

# Cityscapes 标准颜色调色板 (RGB) 
# 对应上面 19 个类别的顺序
CITYSCAPES_PALETTE = [
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],    # building
    [102, 102, 156], # wall
    [190, 153, 153], # fence
    [153, 153, 153], # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],   # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152], # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],   # person
    [255, 0, 0],     # rider
    [0, 0, 142],     # car
    [0, 0, 70],      # truck
    [0, 60, 100],    # bus
    [0, 80, 100],    # train
    [0, 0, 230],     # motorcycle
    [119, 11, 32]    # bicycle
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
# 3. 核心功能函数
# ==========================================
def extract_semantic_features(image_path, return_mask=False):
    """
    输入: 街景图片路径
    输出: 
        - 19维的特征向量 
        - 可选: 语义分割掩膜
    """
    # A. 图像预处理
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"无法读取图片: {e}")
        return None, None, None

    inputs = processor(images=image, return_tensors="pt").to(device)

    # B. 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
        
    # C. 后处理 logits
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], 
        mode="bilinear",
        align_corners=False,
    )

    # D. 生成语义分割掩膜
    # argmax 获取每个像素概率最大的类别索引 (0-18)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # E. 计算像素占比 
    total_pixels = pred_seg.numel()
    pixel_ratios = []
    
    for class_id in range(len(CITYSCAPES_CLASSES)):
        class_pixel_count = (pred_seg == class_id).sum().item()
        ratio = class_pixel_count / total_pixels
        pixel_ratios.append(ratio)
    
    # 是否需要返回掩膜
    if return_mask:
        return np.array(pixel_ratios), pred_seg.cpu().numpy(), image
    else:
        return np.array(pixel_ratios)

def visualize_prediction(pred_seg, original_image, save_path="vis_result.png"):
    """
    将分割掩膜 ID 矩阵转换为彩色图片并与原图叠加
    """
    # 1. 创建彩色掩膜
    # pred_seg 形状: (H, W) -> color_seg 形状: (H, W, 3)
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(CITYSCAPES_PALETTE)
    
    # 使用 numpy 的高级索引，将每个类别ID映射到对应的RGB颜色
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color
        
    # Convert to PIL Image
    color_seg_img = Image.fromarray(color_seg)
    
    # 2. 图像融合 (Overlay)
    # alpha=0.5 表示原图和掩膜各占 50% 透明度
    overlay_img = Image.blend(original_image, color_seg_img, alpha=0.5)
    
    # 3. 保存结果
    overlay_img.save(save_path)
    print(f"可视化结果已保存至: {save_path}")
    
    # 如果想单独保存纯掩膜，取消下面注释
    # color_seg_img.save("mask_only.png")

# ==========================================
# 4. 运行
# ==========================================
if __name__ == "__main__":
    test_image_path = "street_view_sample.png" 
    print(f"正在分析图片: {test_image_path} ...")
    
    # 调用：TRUE请求返回掩膜
    feature_vector, pred_mask, original_img = extract_semantic_features(test_image_path, return_mask=True)
    
    if feature_vector is not None:
        # 1. 打印数据结果
        print("\n=== 实验结果 (Top 元素): ===")
        for i, ratio in enumerate(feature_vector):
            if ratio > 0.001: # 仅显示占比 > 0.1% 的
                print(f"{CITYSCAPES_CLASSES[i].ljust(15)}: {ratio*100:.2f}%")
        
        # 2. 执行可视化
        vis_filename = "vis_" + os.path.basename(test_image_path)
        visualize_prediction(pred_mask, original_img, save_path=vis_filename)