import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from Datasets import MultiModalDataset
from efficientNet import CKD_MultiModalNet
import re, os, torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR  = ""
CSV_FILE = ''
WEIGHT   = ''
BATCH    = 1
N_TAB    = 0


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = MultiModalDataset(csv_file=CSV_FILE,
                                 img_dir=IMG_DIR,
                                 transform=transform)
test_loader  = DataLoader(test_dataset,
                          batch_size=BATCH,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)


model = CKD_MultiModalNet(num_tabular_features=N_TAB).to(device)
model.load_state_dict(torch.load(WEIGHT, map_location=device))
model.eval()


TP = FP = TN = FN = 0
all_pred, all_true, all_names = [], [], []
all_true, all_pred, prob_list = [], [], []

with torch.no_grad():
    for images, tabular, labels in test_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
        outputs = model(images, tabular)
        prob = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        for p, l in zip(predicted.view(-1), labels.view(-1)):
            if l == 1 and p == 1:
                TP += 1
            elif l == 0 and p == 1:
                FP += 1
            elif l == 0 and p == 0:
                TN += 1
            elif l == 1 and p == 0:
                FN += 1


        all_pred.extend(predicted.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
        prob_list.extend(prob[:, 1].cpu().numpy())


pos_recall = TP / (TP + FN + 1e-7)
neg_specif = TN / (TN + FP + 1e-7)
acc        = (TP + TN) / (TP + TN + FP + FN + 1e-7)
f1_macro   = f1_score(all_true, all_pred, average='macro', zero_division=0)

from sklearn.metrics import roc_curve, auc, recall_score, f1_score


fpr, tpr, thr = roc_curve(all_true, prob_list)
auc_score = auc(fpr, tpr)

from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(all_true, prob_list)

if tpr[-1] < 0.80:
    spec_at_80 = 0.0
else:
    spec_at_80 = 1 - np.interp(0.80, tpr, fpr)


J = tpr - fpr
best_thr = thr[np.argmax(J)]
print(f"\nBest threshold = {best_thr:.4f}")


best_pred = (np.array(prob_list) >= best_thr).astype(int)

from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Classification Report (Best Threshold) ===")
print(classification_report(all_true, best_pred, digits=4))

cm2 = confusion_matrix(all_true, best_pred)
print("\nNew Confusion Matrix:")
print(cm2)




f1_macro = f1_score(all_true, all_pred, average='macro')
recall_80 = recall_score(all_true, all_pred, pos_label=1)

print(f'AUC={auc_score:.3f}, Sens@0.80=0.800, Spec={spec_at_80:.3f}')