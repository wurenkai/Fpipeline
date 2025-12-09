import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, recall_score, f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from Datasets import MultiModalDataset
from efficientNet import CKD_MultiModalNet
import torch.nn as nn
import torch.optim as optim

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_CSV    = ''
IMG_DIR     = ""
BATCH       = 32
EPOCHS      = 50
LR          = 6e-4
N_TAB       = 0
WEIGHT_DIR  = ''
os.makedirs(WEIGHT_DIR, exist_ok=True)
# =====================================

# 数据变换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def evaluate(loader, model):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for img, tab, lbl in loader:
            img, tab, lbl = img.to(DEVICE), tab.to(DEVICE), lbl.to(DEVICE)
            out = model(img, tab)
            prob = torch.softmax(out, dim=1)[:, 1]
            y_true.extend(lbl.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    # Sens@0.80
    if tpr[-1] < 0.80:
        spec_at_80 = 0.0
    else:
        spec_at_80 = 1 - np.interp(0.80, tpr, fpr)

    sens_80 = 0.800
    return roc_auc, sens_80, spec_at_80


full_dataset = MultiModalDataset(FULL_CSV, IMG_DIR, transform=transform, scaler_path='scaler.json')
full_idxs    = np.arange(len(full_dataset))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_stats = []

for fold, (train_idx, val_idx) in enumerate(kf.split(full_idxs)):
    print(f'\n===== Fold {fold+1}/5 =====')

    train_fold = Subset(full_dataset, train_idx)
    val_fold   = Subset(full_dataset, val_idx)

    def get_labels(subset):
        labels = []
        for i in range(len(subset)):
            _, _, lbl = subset[i]
            labels.append(lbl)
        return np.array(labels, dtype=np.int64)

    train_labels = get_labels(train_fold)
    class_count  = np.bincount(train_labels)
    weight_per_sample = 1.0 / class_count[train_labels]
    sampler = WeightedRandomSampler(weight_per_sample,
                                    num_samples=len(weight_per_sample),
                                    replacement=True)
    train_loader = DataLoader(train_fold, batch_size=BATCH, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_fold,   batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = CKD_MultiModalNet(
        num_tabular_features=0,
        pretrained_path='',
        device='cuda'
    )
    model.to('cuda')


    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)


    best_val_auc = 0.0
    for epoch in range(EPOCHS):

        model.train()
        for img, tab, lbl in train_loader:
            img, tab, lbl = img.to(DEVICE), tab.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            out = model(img, tab)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

        val_auc, val_sens, val_spec = evaluate(val_loader, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'{WEIGHT_DIR}/fold{fold+1}_best.pth')

    fold_auc, fold_sens, fold_spec = evaluate(val_loader, model)
    fold_stats.append([fold_auc, fold_sens, fold_spec])
    print(f'Fold {fold+1} | AUC={fold_auc:.3f}, Sens@0.80={fold_sens:.3f}, Spec={fold_spec:.3f}')


fold_stats = np.array(fold_stats)
mean, std = fold_stats.mean(axis=0), fold_stats.std(axis=0)



def boot_ci(metric_arr, n_resamples=1000, random_state=42):

    res = bootstrap((metric_arr,), np.mean, n_resamples=n_resamples,
                    confidence_level=0.95, random_state=random_state)
    return res.confidence_interval.low, res.confidence_interval.high

auc_ci   = boot_ci(fold_stats[:, 0])
sens_ci  = boot_ci(fold_stats[:, 1])
spec_ci  = boot_ci(fold_stats[:, 2])

print(f'AUC       : {mean[0]:.3f} ± {std[0]:.3f}  (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})')
print(f'Sens@0.80 : {mean[1]:.3f} ± {std[1]:.3f}  (95% CI {sens_ci[0]:.3f}–{sens_ci[1]:.3f})')
print(f'Spec      : {mean[2]:.3f} ± {std[2]:.3f}  (95% CI {spec_ci[0]:.3f}–{spec_ci[1]:.3f})')