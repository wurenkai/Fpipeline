import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class CKD_MultiModalNet(nn.Module):
    def __init__(self, num_tabular_features=10, pretrained_path=None, device='cuda'):
        super(CKD_MultiModalNet, self).__init__()

        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        image_feat_dim = self.image_model._fc.in_features  # =1536
        self.image_model._fc = nn.Identity()

        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 + num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        if pretrained_path is not None:
            print(f"🔹 Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v

            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Weights loaded. Missing keys: {missing}, Unexpected keys: {unexpected}")

    def forward(self, image, tabular):
        image_feat = self.image_model(image)          # [B, 1536]
        image_feat = self.image_proj(image_feat)      # [B, 16]
        out = self.classifier(image_feat)               # [B, 2]
        return out