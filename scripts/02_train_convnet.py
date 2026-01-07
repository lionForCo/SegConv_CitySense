import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from models.convnext import convnext_base
import torch.nn.functional as F  # 添加导入
import copy
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler  # 添加导入

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch分训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset_size = len(train_loader.dataset)
            else:
                model.eval()
                dataloader = val_loader
                dataset_size = len(val_loader.dataset)

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloader:  
                images = images.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    # outputs = F.softmax(outputs, dim=1)  # 确保输出为概率分布
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                # 保存最优模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
    print(f'Best val Acc: {best_acc:.4f}')
    # 加载验证集最好模型权重
    model.load_state_dict(best_model_wts)
    save_dir = 'bestmodel'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 递归创建目录
    torch.save(best_model_wts, os.path.join(save_dir, 'convnext_best.pth'))

    return model, train_losses, train_accuracies, val_losses, val_accuracies

# 1. 自定义数据集
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)  # 读取标注文件
        self.img_dir = img_dir  # 图片文件夹路径
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从 CSV 文件中获取核心 ID 和标签
        core_id = self.data.iloc[idx, 1]  # Image 列（核心 ID）
        label = int(self.data.iloc[idx, 8])  # Category 列（标签）

        # 根据完整图片路径img_dir/core_id.jpg
        img_name = os.path.join(self.img_dir, f"{core_id}.jpg")  # 构建文件名

        # 打开图片
        # 如果没找到到图片文件，抛出异常
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"无法在目录 {self.img_dir} 中找到文件 '{img_name}'")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 转换标签为 LongTensor
        label = torch.tensor(label).long()

        return image, label

# 2. 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 加载数据
dataframe = pd.read_csv('data/beautiful.csv')

# Divide the training set and test set
train_dataframe = dataframe.sample(frac=0.7, random_state=0)
val_dataframe = dataframe.drop(train_dataframe.index)

train_dataframe.to_csv('dataset/bea_train.csv', index=False)
val_dataframe.to_csv('dataset/bea_val.csv', index=False)

img_dir = "data/data/images"
train_dataset = MyDataset('dataset/bea_train.csv', img_dir, transform=transform_train)
val_dataset = MyDataset('dataset/bea_val.csv', img_dir, transform=transform_val)

# 统计每个类别的样本数
class_counts = train_dataframe['score_category'].value_counts().to_dict()
weights = [1.0 / class_counts[label] for label in train_dataframe['score_category']]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# 划分训练集和验证集
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 4. 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.cuda.empty_cache()
model = convnext_base(pretrained=True, num_classes=10)  # 设置输出类别为10
model = model.to(device)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)



trained_model, train_losses, train_accs, val_losses, val_accs = train_model(
    model, criterion, optimizer, scheduler, num_epochs=50)
torch.cuda.empty_cache()

# Figure
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.savefig("Validation_pair_loss_fy", dpi=900)
plt.show()

plt.figure()
train_accuracies = [x.cpu().numpy() for x in train_accs]
plt.plot(train_accuracies, label='Train Accuracy')
val_accuracies = [x.cpu().numpy() for x in val_accs]
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.savefig("Validation_pair_acc_fy", dpi=900)
plt.show()