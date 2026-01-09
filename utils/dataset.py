import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class PlacePulseDataset(Dataset):
    def __init__(self, csv_file, img_root_dir, transform=None, mode='train'):
        """
        Args:
            csv_file: 指标文件的路径
            img_root_dir: 图片的目录路径
            transform: 图像的变换
            mode: 'train' 或 'inference', inference 不加载标签
            # balance (bool): 是否执行数据平衡（仅在训练集使用）
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.mode = mode

        # 数据的清洗(看数据量)
        self.data_frame = self.data_frame[self.data_frame['location_id'].apply(
           lambda x: os.path.exists(os.path.join(img_root_dir, x + '.jpg')))]
        
        # # 2. 数据平衡 - 训练要求
        # if balance and mode != 'inference':
        #     if 'score_category' in self.data_frame.columns:
        #         g = self.data_frame.groupby('score_category')
        #         # 策略：找到样本数最少的类别数量，然后让所有类别都采样到这个数量
        #         min_count = g.size().min()
        #         # 如果某个类样本太少（比如少于50），可能会导致数据浪费，建议设置一个下限
        #         # 这里为了完全平衡，使用 min_count
        #         print(f"Balancing data: Downsampling all classes to {min_count} samples.")
        #         self.data_frame = g.apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
        #     else:
        #         print("Warning: 'score_category' not found, skipping balance.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imglocation_id = str(self.data_frame.iloc[idx]['location_id'])
        img_name = os.path.join(self.img_root_dir, imglocation_id + '.jpg')

        try:
            image = Image.open(img_name).convert('RGB')
        except (OSError, FileNotFoundError):
            # 图片损坏或找不到，生成一张全黑图避免崩溃
            print(f"Warning: Could not open {img_name}, using black image.")
            image = Image.new('RGB', (384, 384), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'id': imglocation_id}

        # 训练模式读取 Label
        if self.mode != 'inference':
            label = int(self.data_frame.iloc[idx]['score_category'])
            sample['label'] = torch.tensor(label, dtype=torch.long)

        return sample