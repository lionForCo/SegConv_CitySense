import torch
from torchvision import transforms
from PIL import Image
from models.convnext import convnext_base
import os
import torch.nn.functional as F  # 添加导入

# 加载模型
def load_model(model_path, device):
    model = convnext_base(pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 增加批次维度

# 提取特征
def extract_features(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)  # 模型输出 logits
        features = F.softmax(logits, dim=1)  # 转换为概率分布
    return features.cpu().numpy()

if __name__ == "__main__":
    # 配置
    model_path = "bestmodel/convnext_best.pth"  # 模型路径
    image_path = "513d692afdc9f0358700462c.jpg"  # 替换为街景图像路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件未找到: {image_path}")

    # 加载模型
    model = load_model(model_path, device)

    # 预处理图像
    image_tensor = preprocess_image(image_path)

    # 提取特征
    features = extract_features(model, image_tensor, device)

    print("提取的特征:", features)