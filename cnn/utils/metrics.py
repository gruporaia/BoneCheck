import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def compute_metrics(y_true, y_pred, y_probs, num_classes):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        if num_classes == 2:
            # Ensure y_probs has shape (n_samples, 2)
            if len(y_probs.shape) == 2 and y_probs.shape[1] == 2:
                metrics["auc_roc"] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                raise ValueError("Expected y_probs shape (n_samples, 2) for binary classification")
        else:
            metrics["auc_roc"] = roc_auc_score(y_true, y_probs, multi_class="ovr")
    except Exception as e:
        metrics["auc_roc"] = -1  # Error captured, invalid ROC AUC

    return metrics
