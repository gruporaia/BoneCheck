import os
import time
import json
import torch
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from sklearn.utils.class_weight import compute_class_weight

from cnn.utils.datamodule import DataModule
from cnn.utils.metrics import compute_metrics
from cnn.utils.logger import save_fold_outputs

# === Modelos disponíveis ===
from cnn.custom_models.deit import DeiTSmallLightningModel
from cnn.custom_models.efficientnet import EfficientNetV2SClassifier
from cnn.custom_models.swin import SwinTinyLightningModel
from cnn.custom_models.convnext import ConvNeXtLightningModel

# === Use config.constants for all paths and constants ===
from config.constants import DATA_CSV, DATA_DIR, OUTPUTS_DIR, VAL_FOLD, NUM_CLS

torch.manual_seed(42)

model_classes = {
    # "deit": DeiTSmallLightningModel,
    # "swin": SwinTinyLightningModel,
    "efficientnet": EfficientNetV2SClassifier,
    "convnext": ConvNeXtLightningModel,
}

# -------> ESCOLHA O NOME DO MODELO AQUI <---------
model_name = "efficientnet"

def main():
    # === Hiperparâmetros ===
    # param_grid = {
    #     "lr": [1e-4, 1e-5, 1e-6],
    #     "batch_size": [64],
    #     "optimizer": ["Adam"], # Adam para CNNs | AdamW para ViTs
    #     "n_epochs": [5, 30],
    # }
    param_grid = {
        "lr": [1e-4],
        "batch_size": [64],
        "optimizer": ["Adam"], # Adam para CNNs | AdamW para ViTs
        "n_epochs": [100],
    }
    param_combinations = list(product(*param_grid.values()))

    # === Dados ===
    csv_path = DATA_CSV
    image_dir = DATA_DIR
    df = pd.read_csv(csv_path)
    num_classes = df["class_numeric"].nunique()
    
    # Filter for only classes 0 and 2 <-------------- 2 classes
    if NUM_CLS == 2:
        df = df[df['class_numeric'].isin([0, 2])].reset_index(drop=True)
        # Rename class 2 to class 1
        df['class_numeric'] = df['class_numeric'].replace({2: 1})
        num_classes = 2  # Since we only have 2 classes now

    # === K-Folds ===
    df = df[df['fold'] != VAL_FOLD] # tirando os dados de teste
    n_splits = df["fold"].nunique()
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df_train['class_numeric']),
        y=df_train['class_numeric']
    )
    class_weights = torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results_summary = []

    best_auc_global = -1.0
    best_model_info = {}
    for model_name in model_classes.keys():
        for idx, (lr, batch_size, opt_name, n_epochs) in enumerate(param_combinations):
            print(f"\n📊 Combo {idx+1}/{len(param_combinations)}: LR={lr}, BS={batch_size}, OPT={opt_name}, EP={n_epochs}")
            fold_accuracies = []
            fold_aucs = []

            for fold in range(n_splits):
                print(f"\n🔁 Fold {fold + 1}/{n_splits}")
                df_train = df[df["fold"] != fold].reset_index(drop=True)
                df_val = df[df["fold"] == fold].reset_index(drop=True)

                # === DataModule ===
                data_module = DataModule(df_train, df_val, image_dir, batch_size=batch_size)
                data_module.setup()

                # === Modelo ===
                ModelClass = model_classes[model_name]
                model = ModelClass(
                    num_classes=num_classes,
                    lr=lr,
                    optimizer_name=opt_name
                )

                fold_dir = os.path.join(OUTPUTS_DIR, f"{model_name}_fold_{fold}_combo_{idx}")
                os.makedirs(fold_dir, exist_ok=True)

                tb_logger = TensorBoardLogger(
                    save_dir=fold_dir,
                    name="tb_logs"
                )

                early_stopping = pl.callbacks.EarlyStopping(monitor="val_acc", patience=10, mode="max")
                checkpoint_cb = pl.callbacks.ModelCheckpoint(
                    dirpath=fold_dir,
                    filename="best_model",
                    save_top_k=1,
                    monitor="val_acc",
                    mode="max"
                )

                trainer = pl.Trainer(
                    max_epochs=n_epochs,
                    accelerator="cuda" if torch.cuda.is_available() else "cpu",
                    precision="16-mixed",
                    callbacks=[early_stopping, checkpoint_cb],
                    logger=tb_logger,
                    enable_progress_bar=True
                )
                tb_logger.log_hyperparams(model.hparams)

                start_time = time.time()
                trainer.fit(model, data_module)
                train_time = time.time() - start_time

                best_model_path = checkpoint_cb.best_model_path
                best_model = ModelClass.load_from_checkpoint(
                    best_model_path,
                    num_classes=num_classes,
                    lr=lr,
                    optimizer_name=opt_name
                )

                y_true, y_pred, y_probs = best_model.evaluate(data_module.val_dataloader())
                metrics = compute_metrics(y_true, y_pred, y_probs, num_classes=num_classes)
                metrics["train_time_sec"] = train_time
                metrics["hyperparams"] = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "optimizer": opt_name,
                    "n_epochs": n_epochs
                }

                save_fold_outputs(metrics, best_model_path, fold_dir, y_true, y_pred)
                fold_accuracies.append(metrics["accuracy"])
                fold_aucs.append(metrics["auc_roc"])

                # === Verifica se é o melhor AUC global ===
                if metrics["auc_roc"] > best_auc_global:
                    best_auc_global = metrics["auc_roc"]
                    best_model_info = {
                        "model_path": best_model_path,
                        "metrics": metrics,
                        "hyperparams": metrics["hyperparams"]
                    }

            avg_acc = np.mean(fold_accuracies)
            results_summary.append({
                "model": model_name,
                "lr": lr,
                "batch_size": batch_size,
                "optimizer": opt_name,
                "n_epochs": n_epochs,
                "mean_accuracy": avg_acc,
                "mean_auc": np.mean(fold_aucs)
            })

        # === Salva a tabela de resumo ===
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(OUTPUTS_DIR, f"grid_results_{model_name}.csv"), index=False)

        # === Salva o melhor modelo de todos ===
        best_output_dir = os.path.join(OUTPUTS_DIR, f"best_model_{model_name}")
        os.makedirs(best_output_dir, exist_ok=True)

        # Copia o checkpoint
        best_model_ckpt = os.path.join(best_output_dir, f"best_model_{model_name}.ckpt")
        torch.save(torch.load(best_model_info["model_path"]), best_model_ckpt)

        # Salva métricas e hiperparâmetros
        with open(os.path.join(best_output_dir, f"metrics_{model_name}.json"), "w") as f:
            json.dump(best_model_info["metrics"], f, indent=4)

        with open(os.path.join(best_output_dir, f"hyperparams_{model_name}.json"), "w") as f:
            json.dump(best_model_info["hyperparams"], f, indent=4)

        print("\n✅ Finalizado. Resultados salvos em 'outputs/grid_results.csv'")
        print(f"🏆 Melhor modelo salvo em: {best_model_ckpt} com AUC = {best_auc_global:.4f}")

if __name__ == "__main__":
    main()
