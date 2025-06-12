import torch
import torch.nn as nn
from cnn.custom_models.base_model import BaseClassifier

class DeiTSmallLightningModel(BaseClassifier):
    def __init__(self, num_classes: int, lr: float = 1e-3, optimizer_name: str = "AdamW", class_weights=None):
        super().__init__(lr=lr, num_classes=num_classes, optimizer_name=optimizer_name)
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        # for param in self.model.blocks.parameters():
        #     param.requires_grad = False
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)