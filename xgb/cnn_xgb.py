import os
import json
import torch
from PIL import Image
from torchvision import transforms
from cnn.custom_models.deit import DeiTSmallLightningModel
from cnn.custom_models.efficientnet import EfficientNetV2SClassifier 
from cnn.custom_models.swin import SwinTinyLightningModel
from cnn.custom_models.convnext import ConvNeXtLightningModel
from cnn.utils.metrics import compute_metrics
from cnn.utils.datamodule import DataModule
import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.constants import WEIGHTS_DIR, DATA_CSV, DATA_DIR, VAL_FOLD, BATCH_SIZE, XGB_RESULTS_DIR, NUM_CLS

class CNNEnsembleXGBoost:
    def __init__(self, weights_dir, data_module, device=None):
        if not os.path.exists(weights_dir):
            raise ValueError(f"Weights directory '{weights_dir}' does not exist")
            
        self.weights_dir = weights_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_classes = {
            "convnext": ConvNeXtLightningModel,
            "deit": DeiTSmallLightningModel,
            "efficientnet": EfficientNetV2SClassifier,
            "swin": SwinTinyLightningModel
        }
        self.models = self._load_all_models()
        if not self.models:
            raise ValueError(f"No model checkpoints found in '{weights_dir}'")
            
        self.xgb_model = None
        self.data_module = data_module
        
        # Move all models to the specified device
        for model in self.models:
            model.to(self.device)
            model.eval()

    def _find_all_checkpoints(self):
        checkpoints = []
        for model_name in self.model_classes.keys():
            model_dir = os.path.join(self.weights_dir, model_name)
            if not os.path.exists(model_dir):
                continue
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".ckpt"):
                        checkpoints.append((model_name, os.path.join(root, file)))
        return checkpoints

    def _load_checkpoint(self, model_class, checkpoint_path):
        try:
            model = model_class.load_from_checkpoint(checkpoint_path)
            return model
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
            return None

    def _load_all_models(self):
        checkpoints = self._find_all_checkpoints()
        loaded_models = []
        for model_name, ckpt_path in checkpoints:
            model_class = self.model_classes[model_name]
            model = self._load_checkpoint(model_class, ckpt_path)
            if model is not None:
                loaded_models.append(model)
        return loaded_models

    def extract_features(self, image_paths):
        features = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise ValueError(f"Image path does not exist: {img_path}")
                
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.data_module.transform(img).unsqueeze(0).to(self.device)
                img_features = []
                
                with torch.no_grad():
                    for model in self.models:
                        logits = model(img_tensor)
                        probs = torch.softmax(logits, dim=1)
                        img_features.extend(probs.cpu().numpy().flatten())
                        
                features.append(img_features)
            except Exception as e:
                raise RuntimeError(f"Error processing image {img_path}: {str(e)}")
                
        return np.array(features)

    def fit(self, train_image_paths, train_labels, val_image_paths, val_labels, random_state=42):
        # Extract features for both training and validation sets
        print("Extracting features from training images...")
        train_features = self.extract_features(train_image_paths)
        print("Extracting features from validation images...")
        val_features = self.extract_features(val_image_paths)

        # Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model = xgboost.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
        self.xgb_model.fit(train_features, train_labels)

        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_pred = self.xgb_model.predict(val_features)
        val_probs = self.xgb_model.predict_proba(val_features)
        metrics = compute_metrics(val_labels, val_pred, val_probs, len(np.unique(train_labels)))
        self._last_test = (val_features, val_labels, val_pred, val_probs)
        return metrics

    def predict(self, image_paths):
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not trained. Call fit() first.")
        features = self.extract_features(image_paths)
        preds = self.xgb_model.predict(features)
        probs = self.xgb_model.predict_proba(features)
        return preds, probs

    def save_results(self, metrics, save_dir=XGB_RESULTS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        results = {
            "metrics": metrics,
        }
        if hasattr(self, "_last_test"):
            X_test, y_test, y_pred, y_probs = self._last_test
            results["predictions"] = {
                "true": y_test.tolist(),
                "pred": y_pred.tolist(),
                "probs": y_probs.tolist()
            }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    def load_xgb_model(self, model_path):
        """
        Load a trained XGBoost model from the given file path.
        """
        self.xgb_model = xgboost.XGBClassifier()
        self.xgb_model.load_model(model_path)

def main():
    # Load all models from all checkpoints in weights_old
    weights_dir = WEIGHTS_DIR
    if not os.path.exists(weights_dir):
        raise ValueError(f"Weights directory '{weights_dir}' does not exist")

    # === Load data using DataModule, as in train.py ===
    csv_path = DATA_CSV
    image_dir = DATA_DIR
    
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file '{csv_path}' does not exist")
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory '{image_dir}' does not exist")
        
    df = pd.read_csv(csv_path)
    num_classes = df["class_numeric"].nunique()

    # Set which fold to use as validation
    val_fold = VAL_FOLD

    if NUM_CLS == 2:
        df = df[df['class_numeric'].isin([0, 2])].reset_index(drop=True)
        # Rename class 2 to class 1
        df['class_numeric'] = df['class_numeric'].replace({2: 1})
        num_classes = 2  # Since we only have 2 classes now
    
    if "fold" not in df.columns:
        raise ValueError("No 'fold' column found in CSV for validation split.")
        
    df_val = df[df['fold'] == val_fold].reset_index(drop=True)
    df_train = df[df['fold'] != val_fold].reset_index(drop=True)
    print('\n\n\n\n\n')
    print('\n\n\n\n\n')
    print(df_train)
    # Validate that we have some validation data
    if len(df_val) == 0:
        raise ValueError(f"No validation data found for fold {val_fold}")

    # Use DataModule to get val set (for XGBoost training)
    batch_size = BATCH_SIZE
    data_module = DataModule(df_train, df_val, image_dir, batch_size=batch_size)
    print(' ---- LEN -------------- ', len(data_module.df_train))
    data_module.setup()

    # Validate that validation dataset has samples
    if len(data_module.val_dataset) == 0:
        print("Checking image directory structure...")
        # Check if any validation fold directory exists
        val_fold_dir = os.path.join(image_dir, f"fold_{val_fold}")
        if not os.path.exists(val_fold_dir):
            raise ValueError(f"Validation fold directory not found: {val_fold_dir}")
        
        # Check if images exist with expected suffixes
        sample_path = df_val["path"].iloc[0] if len(df_val) > 0 else None
        if sample_path:
            file_id = os.path.splitext(sample_path)[0]
            expected_path = os.path.join(val_fold_dir, f"{file_id}_cropped.png")
            print(f"Expected image path: {expected_path}")
            if not os.path.exists(expected_path):
                raise ValueError("No images found with expected naming pattern. Images should be named like: <file_id>_cropped_bottom_right_bright.png")
        raise ValueError("No samples found in validation dataset. Please check image paths and directory structure.")

    ensemble = CNNEnsembleXGBoost(weights_dir, data_module)
    print(f"Loaded {len(ensemble.models)} models")

    # Get training images and labels
    train_dataset = data_module.train_dataset
    train_image_paths = []
    train_labels = []
    for img_path, label in train_dataset.samples:
        train_image_paths.append(img_path)
        train_labels.append(label)
    train_labels = np.array(train_labels)

    # Get validation images and labels
    val_dataset = data_module.val_dataset
    val_image_paths = []
    val_labels = []
    for img_path, label in val_dataset.samples:
        val_image_paths.append(img_path)
        val_labels.append(label)
    val_labels = np.array(val_labels)

    print(f"Found {len(train_image_paths)} training images")
    print(f"Found {len(val_image_paths)} validation images")

    # Train XGBoost and get metrics on validation set
    metrics = ensemble.fit(train_image_paths, train_labels, val_image_paths, val_labels)
    print("\nXGBoost Ensemble Results on Validation Set:")
    print("-------------------------------------------")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value:.4f}")

    # Save results
    ensemble.save_results(metrics)

    # Example: Predict on new images (here, just reuse first 10 from val set for demonstration)
    # preds, probs = ensemble.predict(val_image_paths)
    # print("Predictions for first 10 images:", preds)
    # print("Probabilities for first 10 images:", probs)

if __name__ == "__main__":
    main()