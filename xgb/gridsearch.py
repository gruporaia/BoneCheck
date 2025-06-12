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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgb.cnn_xgb import CNNEnsembleXGBoost
from config.constants import DATA_CSV, DATA_DIR, WEIGHTS_DIR, XGBOOST_DIR, VAL_FOLD
import joblib

def main():
    # Use constants for paths
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

    if "fold" not in df.columns:
        raise ValueError("No 'fold' column found in CSV for validation split.")
        
    df_val = df[df['fold'] == val_fold].reset_index(drop=True)
    df_train = df[df['fold'] != val_fold].reset_index(drop=True)



    df = df[df['class_numeric'].isin([0, 2])].reset_index(drop=True)
    # Rename class 2 to class 1
    df['class_numeric'] = df['class_numeric'].replace({2: 1})
    # Set which fold to use as validation


    # Validate that we have some validation data
    if len(df_val) == 0:
        raise ValueError(f"No validation data found for fold {val_fold}")

    # Use DataModule to get val set (for XGBoost training)
    batch_size = 64
    data_module = DataModule(df_train, df_val, image_dir, batch_size=batch_size)
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
            expected_path = os.path.join(val_fold_dir, f"{file_id}_cropped_bottom_right_bright.png")
            print(f"Expected image path: {expected_path}")
            if not os.path.exists(expected_path):
                raise ValueError("No images found with expected naming pattern. Images should be named like: <file_id>_cropped_bottom_right_bright.png")
        raise ValueError("No samples found in validation dataset. Please check image paths and directory structure.")

    ensemble = CNNEnsembleXGBoost(weights_dir, data_module)
    print(f"Loaded {len(ensemble.models)} models")

    # Get all validation images and labels
    val_dataset = data_module.val_dataset
    image_paths = []
    labels = []
    for img_path, label in val_dataset.samples:
        image_paths.append(img_path)
        labels.append(label)
    labels = np.array(labels)

    print(f"Found {len(image_paths)} validation images")

    # === HYPERPARAMETER SEARCH ONLY BELOW ===

    # Extract features using the ensemble
    features = ensemble.extract_features(image_paths)

    # Define parameter grid for XGBoost
    param_grid = {
        'n_estimators': [25],
        'max_depth': [2],
        'learning_rate': [0.02],
        'min_child_weight': [1],
        'gamma': [1],
        'subsample': [0.7],
        'colsample_bytree': [0.85]
    }
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'auc_roc': make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }

    # Initialize base XGBoost model
    base_model = xgboost.XGBClassifier(
        objective='multi:softproba',
        random_state=42,
        tree_method='hist',
        gpu_id=0
    )

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        refit='auc_roc',  # Optimize for AUC-ROC
        cv=5,
        n_jobs=-1,  # Use all available CPU cores
        verbose=2
    )

    print("Starting hyperparameter search...")
    grid_search.fit(features, labels)

    # Save the best model
    os.makedirs(XGBOOST_DIR, exist_ok=True)
    joblib.dump(grid_search.best_estimator_, os.path.join(XGBOOST_DIR, "xgb_model.json"))

    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
        
    print("\nBest cross-validation scores:")
    for metric in scoring.keys():
        mean_scores = grid_search.cv_results_['mean_test_' + metric]
        print(f"{metric}: {mean_scores[grid_search.best_index_]:.4f}")

    # Save results
    results = {
        "best_params": grid_search.best_params_,
        "cv_results": {
            metric: scores for metric, scores in grid_search.cv_results_.items()
            if metric.startswith('mean_test_') or metric.startswith('std_test_')
        }
    }
    os.makedirs(XGBOOST_DIR, exist_ok=True)
    with open(os.path.join(XGBOOST_DIR, "hyperparam_search_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()


# Best parameters found:
# colsample_bytree: 1.0
# gamma: 0.2
# learning_rate: 0.01
# max_depth: 3
# min_child_weight: 1
# n_estimators: 100
# subsample: 0.8