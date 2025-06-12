import os

# FOLDERS
DATA_DIR = os.getenv("DATA_DIR", "./data/images")
DATA_CSV = os.getenv("DATA_CSV", "./data/data_with_folds.csv")
MODELS_DIR = os.getenv("MODELS_DIR", "./cnn/models")
LOGS_DIR = os.getenv("LOGS_DIR", "./cnn/logs")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "./cnn/outputs")
XGBOOST_DIR = os.getenv("XGBOOST_DIR", "./xgb")
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "./cnn/weights")
XGB_RESULTS_DIR = os.getenv("XGB_RESULTS_DIR", "./xgb/results")
NUM_CLS = os.getenv("NUM_CLS", 3)

VAL_FOLD = os.getenv("VAL_FOLD", -1)

# HYPERPARAMETERS
BATCH_SIZE = os.getenv("BATCH_SIZE", 64)
LR = os.getenv("LR", 1e-4)

