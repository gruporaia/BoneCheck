import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, MulticlassAUROC

class BaseClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, optimizer_name: str = "Adam", class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auc_roc = MulticlassAUROC(num_classes=num_classes, average="macro")

        # To accumulate outputs for AUC at epoch end
        self.validation_probs = []
        self.validation_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.hparams.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f"Otimizador '{self.hparams.optimizer_name}' não suportado")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        
        # Accumulate for AUC calculation later
        self.validation_probs.append(probs)
        self.validation_targets.append(y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        probs = torch.cat(self.validation_probs, dim=0)
        targets = torch.cat(self.validation_targets, dim=0)
        # Compute and log the AUC
        auc = self.val_auc_roc(probs, targets)
        self.log("val_auc_roc", auc, prog_bar=True)

        # Reset for next epoch
        self.validation_probs.clear()
        self.validation_targets.clear()

    def evaluate(self, dataloader):
        self.eval()
        y_true, y_pred, y_probs = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_probs.extend(probs.cpu().tolist())

        return y_true, y_pred, y_probs
