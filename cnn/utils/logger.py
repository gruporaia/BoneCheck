import json
import pandas as pd
import os

def save_fold_outputs(metrics, ckpt_path, output_dir, y_true, y_pred):
    # Salvar métricas
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Salvar predições
    df_preds = pd.DataFrame({
        "true": y_true,
        "pred": y_pred
    })
    df_preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Salvar caminho do modelo
    with open(os.path.join(output_dir, "checkpoint_path.txt"), "w") as f:
        f.write(ckpt_path)
