from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from cnn.custom_models.base_model import BaseClassifier
import torch.nn as nn

class SwinTinyLightningModel(BaseClassifier):
    def __init__(self, num_classes: int, lr: float = 1e-3, optimizer_name: str = "AdamW", class_weights=None):
        super().__init__(lr=lr, num_classes=num_classes, optimizer_name=optimizer_name)
        weights = Swin_V2_T_Weights.IMAGENET1K_V1
        self.model = swin_v2_t(weights=weights)
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)