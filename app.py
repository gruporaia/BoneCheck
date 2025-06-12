from xgb.cnn_xgb import CNNEnsembleXGBoost
from cnn.utils.datamodule import DataModule
import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from config.constants import DATA_CSV, DATA_DIR, WEIGHTS_DIR, XGB_RESULTS_DIR

st.set_page_config(page_title="CNN XGBoost Ensemble", layout="wide")

st.title("CNN XGBoost Ensemble Demo")

@st.cache_resource
def load_ensemble(weights_dir, _data_module):
    return CNNEnsembleXGBoost(weights_dir, _data_module)

@st.cache_data
def load_data(csv_path):
    try:
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found at: {csv_path}")
            st.stop()
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        st.stop()

def get_image_paths_and_labels(val_dataset):
    image_paths = []
    labels = []
    for img_path, label in val_dataset.samples:
        image_paths.append(img_path)
        labels.append(label)
    return image_paths, np.array(labels)

# Add label mapping
LABEL_MAP = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporose"
}

def main():
    st.sidebar.header("Configuration")
    weights_dir = st.sidebar.text_input("Weights Directory", WEIGHTS_DIR)
    csv_path = st.sidebar.text_input("CSV Path", DATA_CSV)
    image_dir = st.sidebar.text_input("Image Directory", DATA_DIR)
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, value=64, step=1)
    val_fold = st.sidebar.number_input("Validation Fold", min_value=0, value=1, step=1)
    xgb_model_path = st.sidebar.text_input("XGBoost Model Path", os.path.join(XGB_RESULTS_DIR, "xgb_model.json"))

    st.write("## 1. Load Ensemble, Data, and XGBoost Model")
    with st.spinner("Loading ensemble, data, and XGBoost model..."):
        df = load_data(csv_path)
        num_classes = df["class_numeric"].nunique()
        if "fold" in df.columns:
            df = df[df['fold'] != -1].reset_index(drop=True)
        n_splits = df["fold"].nunique() if "fold" in df.columns else 5
        df_train = df[df["fold"] != val_fold].reset_index(drop=True)
        df_val = df[df["fold"] == val_fold].reset_index(drop=True)
        data_module = DataModule(df_train, df_val, image_dir, batch_size=batch_size)
        data_module.setup()
        
        # Initialize ensemble with data_module
        ensemble = load_ensemble(weights_dir, data_module)
        st.success(f"Loaded {len(ensemble.models)} models from {weights_dir}")
        
        val_dataset = data_module.val_dataset
        image_paths, labels = get_image_paths_and_labels(val_dataset)
        # Load the trained XGBoost model
        try:
            ensemble.load_xgb_model(xgb_model_path)
            st.success(f"Loaded XGBoost model from {xgb_model_path}")
        except Exception as e:
            st.error(f"Failed to load XGBoost model: {e}")
            st.stop()

    st.write("## 2. Predict on Validation Images")
    if st.button("Predict on Validation Images"):
        with st.spinner("Predicting..."):
            preds, probs = ensemble.predict(image_paths)
        # Map predicted and true labels to names
        pred_labels_named = [LABEL_MAP.get(p, str(p)) for p in preds[:10]]
        true_labels_named = [LABEL_MAP.get(l, str(l)) for l in labels[:10]]
        st.write("### Predictions (first 10)")
        st.write(pd.DataFrame({
            "image_path": image_paths[:10],
            "true_label": true_labels_named,
            "pred_label": pred_labels_named
        }))
        st.write("### Probabilities (first 10)")
        st.write(probs[:10])

    st.write("## 3. Upload and Predict on Your Own Images")
    uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        user_image_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            user_image_paths.append(file_path)
        with st.spinner("Predicting on uploaded images..."):
            preds, probs = ensemble.predict(user_image_paths)
        st.write("### Uploaded Image Predictions")
        for i, img_path in enumerate(user_image_paths):
            label_name = LABEL_MAP.get(preds[i], str(preds[i]))
            st.image(img_path, caption=f"Pred: {label_name}, Prob: {probs[i].max():.2f}", width=200)

if __name__ == "__main__":
    main()