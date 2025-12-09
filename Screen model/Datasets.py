from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch
import json

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, scaler_path='scaler.json'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        with open(scaler_path, 'r') as f:
            s = json.load(f)
        self.vasc_mean = torch.tensor([s['mean']['UWF_Df'],  s['mean']['UWF_TORT']])
        self.vasc_std  = torch.tensor([s['std']['UWF_Df'],   s['std']['UWF_TORT']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor(row[['sex', 'age', 'diabetes', 'hypertension',
                                     'stroke', 'coronary', 'dyslipidemia']].values.astype(float),
                                dtype=torch.float32)

        vascular = torch.tensor(row[['UWF_Df', 'UWF_TORT']].values.astype(float),
                                dtype=torch.float32)
        vascular = (vascular - self.vasc_mean) / self.vasc_std

        tabular_features = torch.cat([clinical, vascular])
        label = int(row['label'])
        return image, tabular_features, label