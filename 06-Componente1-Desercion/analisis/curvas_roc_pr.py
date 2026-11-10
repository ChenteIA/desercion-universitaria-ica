"""Curvas ROC y precision-exhaustividad de los 3 modelos en la validacion de Ica.

Reconstruye la Figura referenciada en Capitulo-III como curvas_roc_pr_ica.png,
que no tenia script de regeneracion propio en el repo (se genero originalmente
de forma ad-hoc). Reutiliza exactamente la misma logica de carga y coercion de
X_ica/y_ica que src/evaluar.py, sin modificar ese archivo.

Anclas: Ica 190 casos / 42 desertores; AUC-ROC rf=0.716 lr=0.663 xgb=0.699.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                               roc_auc_score, roc_curve)

TESIS = Path(__file__).parent.parent.parent
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
from entrenar import CATEGORICAS  # noqa: E402

CSV = TESIS / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARS = TESIS / "05-Datos" / "procesados" / "variables_seleccionadas.json"
FIGURAS = TESIS / "08-Resultados" / "Figuras" / "productivo"

MODELOS = {
    "rf": ("Random Forest", "tab:blue", "-"),
    "lr": ("Regresión logística", "tab:orange", "--"),
    "xgb": ("XGBoost", "tab:green", "-."),
}


def cargar_ica():
    df = pd.read_csv(CSV)
    df["es_ica"] = df["es_ica"].astype(bool)
    variables = json.load(open(VARS, encoding="utf-8"))["seleccionadas_finales"]
    ica = df[df["es_ica"]]
    x_ica, y_ica = ica[variables].copy(), ica["target"].astype(int)
    num_vars = [v for v in variables if v not in CATEGORICAS]
    for col in num_vars:
        x_ica[col] = pd.to_numeric(x_ica[col], errors="coerce")
    assert len(x_ica) == 190 and int(y_ica.sum()) == 42, "Ica no cuadra"
    return x_ica, y_ica


def main():
    x_ica, y_ica = cargar_ica()
    prevalencia = y_ica.mean()

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    for nombre, (etiqueta, color, estilo) in MODELOS.items():
        pipe = joblib.load(REPO / "modelos" / f"pipeline_{nombre}.pkl")
        proba = pipe.predict_proba(x_ica)[:, 1]

        fpr, tpr, _ = roc_curve(y_ica, proba)
        auc_roc = roc_auc_score(y_ica, proba)
        ax_roc.plot(fpr, tpr, color=color, linestyle=estilo,
                    label=f"{etiqueta} (AUC = {auc_roc:.3f})")

        precision, recall, _ = precision_recall_curve(y_ica, proba)
        auc_pr = average_precision_score(y_ica, proba)
        ax_pr.plot(recall, precision, color=color, linestyle=estilo,
                   label=f"{etiqueta} (AUC-PR = {auc_pr:.3f})")

    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Sensibilidad")
    ax_roc.set_title("Curvas ROC (Ica)")
    ax_roc.legend(loc="lower right")

    ax_pr.axhline(prevalencia, color="gray", linestyle=":", linewidth=1)
    ax_pr.text(0.02, prevalencia + 0.02, f"prevalencia = {prevalencia:.3f}",
               color="gray", fontsize=9)
    ax_pr.set_xlabel("Sensibilidad (exhaustividad)")
    ax_pr.set_ylabel("Precisión")
    ax_pr.set_title("Curvas precisión-exhaustividad (Ica)")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURAS / "curvas_roc_pr_ica.png", dpi=150)
    plt.close()
    print("OK curvas_roc_pr_ica.png")


if __name__ == "__main__":
    main()
