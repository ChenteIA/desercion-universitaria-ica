"""Tabla de importancia SHAP global en la validacion de Ica (los valores detras
de la Figura 11 / shap_global_ica.png), para graficarla en las diapositivas.

Reutiliza el pipeline_rf.pkl productivo (no lo reentrena) y el mismo
TreeExplainer de src/interpretar.py sobre las 190 observaciones de Ica. Por
cada columna transformada reporta la media de |SHAP| (el orden del summary
plot) y la direccion del efecto: la correlacion de Spearman entre el valor de
la variable y su valor SHAP (+ = valores altos elevan el riesgo).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from entrenar import CATEGORICAS
from interpretar import calcular_shap

MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
OUT_DIR = BASE_DIR / "analisis" / "salidas"
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]
    num_vars = [v for v in variables if v not in CATEGORICAS]

    ica = df[df["es_ica"]]
    X = ica[variables].copy()
    for col in num_vars:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    pipeline = joblib.load(MODELOS_DIR / "pipeline_rf.pkl")
    preprocesamiento = pipeline.named_steps["preprocesamiento"]
    X_t = preprocesamiento.transform(X)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
    nombres = list(preprocesamiento.get_feature_names_out())
    shap_values = calcular_shap(pipeline.named_steps["modelo"], X_t)

    filas = []
    for j, nombre in enumerate(nombres):
        valores, contrib = pd.Series(X_t[:, j]), pd.Series(shap_values[:, j])
        rho = valores.corr(contrib, method="spearman") if valores.nunique() > 1 else float("nan")
        filas.append({"variable": nombre,
                      "shap_medio_abs": float(np.abs(shap_values[:, j]).mean()),
                      "direccion_spearman": None if pd.isna(rho) else float(rho)})
    ranking = sorted(filas, key=lambda f: f["shap_medio_abs"], reverse=True)

    for f in ranking[:15]:
        print(f"{f['variable']:<40} |SHAP| medio = {f['shap_medio_abs']:.4f}   rho = {f['direccion_spearman']}")
    with open(OUT_DIR / "shap_global_ica.json", "w", encoding="utf-8") as fh:
        json.dump({"n": int(len(ica)), "ranking": ranking}, fh, ensure_ascii=False, indent=2)
    print("\nOK ->", OUT_DIR / "shap_global_ica.json")


if __name__ == "__main__":
    main()
