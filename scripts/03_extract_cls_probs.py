import os
import sys
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.convnext import convnext_base
from utils.dataset import PlacePulseDataset

METRICS = ['beautiful', 'boring', 'depressing', 'lively', 'safe', 'wealthy']
IMG_SIZE = 384
BATCH_SIZE = 64
NUM_CLASSES = 10

def extract_probs(metric, args):
    print(f"\nExtracting probabilities for: {metric}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径
    csv_path = os.path.join(args.data_root, f"{metric}.csv")
    img_dir = os.path.join(args.data_root, "data")
    model_path = os.path.join(args.model_dir, f"best_model_{metric}.pth")
    output_csv = os.path.join(args.output_dir, f"{metric}_probs.csv")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}, skipping.")
        return

    # 数据加载 (无增强)
    transform = transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.14), int(IMG_SIZE * 1.14))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 使用 mode='inference'，不返回 label
    dataset = PlacePulseDataset(csv_path, img_dir, transform=transform, mode='train') # 这里用train模式为了读取ID和标签对比，或者用inference也可以
    # 为了方便后续对齐，我们最好保留原始的 location_id
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 加载模型
    model = convnext_base(pretrained=False, num_classes=NUM_CLASSES)
    # 加载权重 (注意处理 state_dict 键值匹配)
    checkpoint = torch.load(model_path, map_location=device)
    msg = model.load_state_dict(checkpoint, strict=True) 
    print(f"Model loaded. {msg}") # 打印加载信息，确认没有 missing keys
    model = model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            imglocation_ids = batch['id']
            # 如果你也想保存真实标签用于后续对比
            labels = batch['label'].cpu().numpy() if 'label' in batch else [None]*len(imglocation_ids)
            
            # 推理
            logits = model(images)
            # 核心：计算 Softmax 概率
            probs = F.softmax(logits, dim=1) # [Batch, 10]
            
            probs_np = probs.cpu().numpy()
            
            for i in range(len(imglocation_ids)):
                row = {
                    'location_id': imglocation_ids[i],
                    'true_category': labels[i]
                }
                # 保存 class_0 到 class_9 的概率
                for c in range(NUM_CLASSES):
                    row[f'prob_{c}'] = probs_np[i, c]
                results.append(row)
    
    # 保存结果
    df_res = pd.DataFrame(results)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df_res.to_csv(output_csv, index=False)
    print(f"Saved probabilities to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw", help="Path to raw csvs and data folder")
    parser.add_argument("--model_dir", type=str, default="models/bestmodel", help="Directory with .pth models")
    parser.add_argument("--output_dir", type=str, default="data/processed_probs", help="Directory to save probability csvs")
    parser.add_argument("--metric", type=str, default="all")
    args = parser.parse_args()

    if args.metric == 'all':
        for m in METRICS:
            extract_probs(m, args)
    else:
        extract_probs(args.metric, args)