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
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.mode = mode

        # 数据的清洗
        self.data_frame = self.data_frame[self.data_frame['_id'].apply(
           lambda x: os.path.exists(os.path.join(img_root_dir, x + '.jpg')))]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = str(self.data_frame.iloc[idx]['_id'])
        img_name = os.path.join(self.img_root_dir, img_id + '.jpg')

        try:
            image = Image.open(img_name).convert('RGB')
        except (OSError, FileNotFoundError):
            # 图片损坏或找不到，生成一张全黑图避免崩溃
            print(f"Warning: Could not open {img_name}, using black image.")
            image = Image.new('RGB', (384, 384), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'id': img_id}

        # 训练模式读取 Label
        if self.mode != 'inference':
            label = int(self.data_frame.iloc[idx]['score_category'])
            sample['label'] = torch.tensor(label, dtype=torch.long)

        return sample