import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd

# 将项目根目录加入 path 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.convnext import convnext_base 
from utils.dataset import PlacePulseDataset

# 配置
METRICS = ['beautiful', 'boring', 'depressing', 'lively', 'safe', 'wealthy']
IMG_SIZE = 384  # ConvNeXt 推荐分辨率
BATCH_SIZE = 16 # 根据显存调整
EPOCHS = 100     # 根据需要调整
LEARNING_RATE = 1e-4 # 微调通常使用较小 LR
NUM_CLASSES = 10

def balance_dataframe(df, min_samples=None):
    """
    对 DataFrame 进行欠采样平衡。
    """
    # 统计各类别数量
    counts = df['score_category'].value_counts()
    print("原始分布:\n", counts.sort_index())
    
    if min_samples is None:
        # 设置目标数量       
        target_count = 1500 
    else:
        target_count = min_samples

    balanced_df = pd.DataFrame()
    for label in range(10):
        df_class = df[df['score_category'] == label]
        if len(df_class) > target_count:
            # 随机采样，只取 target_count 个
            df_class = df_class.sample(target_count, random_state=42)
        balanced_df = pd.concat([balanced_df, df_class])
        
    # 打乱顺序
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("平衡后分布:\n", balanced_df['score_category'].value_counts().sort_index())
    return balanced_df


def train_one_metric(metric, args):
    print(f"\n{'='*100} Start Training Metric: {metric} {'='*100}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 准备数据
    csv_path = os.path.join(args.data_root, f"{metric}.csv")
    img_dir = os.path.join(args.data_root, "images/")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Skipping.")
        return

    # 数据增强与预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),                        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)                       
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((int(IMG_SIZE * 1.14), int(IMG_SIZE * 1.14))), 
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 加载完整数据
    full_dataset_raw = PlacePulseDataset(csv_path, img_dir, transform=None)
    full_df = full_dataset_raw.data_frame
    
    # 数据平衡
    balanced_df = balance_dataframe(full_df, min_samples=2000) 
    
    # 重新封装为 Dataset
    full_dataset = PlacePulseDataset(csv_path, img_dir, transform=train_transform)
    full_dataset.data_frame = balanced_df 
    
    # 划分数据集 7:2:1 
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    print(f"Total Balanced Samples: {total_size}")
    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")

    # 4. 直接切分 balanced_df
    train_df = balanced_df.iloc[:train_size].reset_index(drop=True)
    val_df = balanced_df.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_df = balanced_df.iloc[train_size + val_size:].reset_index(drop=True)

    # 5. 实例化 Dataset 
    train_ds = PlacePulseDataset(csv_path, img_dir, transform=train_transform)
    train_ds.data_frame = train_df 

    val_ds = PlacePulseDataset(csv_path, img_dir, transform=val_transform)
    val_ds.data_frame = val_df   

    test_ds = PlacePulseDataset(csv_path, img_dir, transform=val_transform) 
    test_ds.data_frame = test_df

    # 6. 检查标签分布
    print(f"Train Dataset Real Length: {len(train_ds)}")
    print(f"Val Dataset Real Length: {len(val_ds)}")
    print("Train Label Distribution check:", train_ds.data_frame['score_category'].value_counts().sort_index().to_dict())

    # 7. DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 初始化模型
    # 使用convnext_base, ImageNet-1k 权重
    model = convnext_base(pretrained=True, in_22k=False, num_classes=NUM_CLASSES, drop_path_rate=0.5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. 训练循环
    best_acc = 0.0
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, f"best_model_{metric}.pth")

    # 早停
    early_stop_counter = 0
    patience = 20  # 如果20个epoch验证集精度不提升，就停止

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        scheduler.step()

        # 验证
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': val_running_loss / val_total,
                    'acc': val_correct / val_total
                })

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            early_stop_counter = 0 # 重置早停
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs.")

        # 早停
        if early_stop_counter >= patience:
            print(f"Early stopping triggered. Best Val Acc was {best_acc:.4f}")
            break

    print(f"Finished {metric}. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw", help="Path to raw csvs and data folder")
    parser.add_argument("--save_dir", type=str, default="models/bestmodel", help="Path to save models")
    parser.add_argument("--metric", type=str, default="all", help="Specific metric to train or 'all'")
    args = parser.parse_args()

    if args.metric == 'all':
        for m in METRICS:
            train_one_metric(m, args)
    else:
        if args.metric in METRICS:
            train_one_metric(args.metric, args)
        else:
            print(f"Invalid metric. Choose from {METRICS}")