from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from cnn.custom_models.base_model import BaseClassifier
import torch.nn as nn
import pytorch_lightning as pl

class ConvNeXtLightningModel(BaseClassifier, pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, optimizer_name: str = "AdamW", class_weights=None, **kwargs):
        super().__init__(lr=lr, num_classes=num_classes, optimizer_name=optimizer_name)
        self.save_hyperparameters()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.model = convnext_tiny(weights=weights)
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        # Use os pesos na loss, se fornecidos
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
