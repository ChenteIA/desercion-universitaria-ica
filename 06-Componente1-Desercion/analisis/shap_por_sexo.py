"""Calcula SHAP por separado para hombres y mujeres en la validacion de Ica,
para verificar empiricamente la hipotesis planteada en 4.5 del Informe: que
la brecha de equidad por sexo (AUC 0.805 hombres vs. 0.676 mujeres) se debe
a una interaccion entre el sexo y el resto de los predictores que el modelo
no captura por igual en ambos subgrupos.

Reutiliza el pipeline_rf.pkl productivo (no lo reentrena) y el mismo
TreeExplainer de src/interpretar.py, aplicado por separado a cada subgrupo
de Ica (P207: 1=hombre, 2=mujer). Compara el ranking de importancia media
|SHAP| entre ambos subgrupos.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from entrenar import CATEGORICAS
from interpretar import calcular_shap

MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
OUT_DIR = BASE_DIR / "analisis" / "salidas"
OUT_DIR.mkdir(exist_ok=True)


def importancia_media(shap_values: np.ndarray, nombres: list) -> pd.DataFrame:
    return (pd.DataFrame({"variable": nombres, "shap_medio_abs": np.abs(shap_values).mean(axis=0)})
            .sort_values("shap_medio_abs", ascending=False)
            .reset_index(drop=True))


def main():
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]
    num_vars = [v for v in variables if v not in CATEGORICAS]

    ica = df[df["es_ica"]]
    pipeline = joblib.load(MODELOS_DIR / "pipeline_rf.pkl")
    modelo_rf = pipeline.named_steps["modelo"]
    preprocesamiento = pipeline.named_steps["preprocesamiento"]

    resultados = {}
    rankings = {}
    for etiqueta, codigo in [("hombres", 1), ("mujeres", 2)]:
        sub = ica[ica["P207"] == codigo]
        X = sub[variables].copy()
        for col in num_vars:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X_t = preprocesamiento.transform(X)
        nombres = preprocesamiento.get_feature_names_out()
        shap_values = calcular_shap(modelo_rf, X_t)
        ranking = importancia_media(shap_values, list(nombres))
        rankings[etiqueta] = ranking
        resultados[etiqueta] = {"n": int(len(sub)), "top10": ranking.head(10).to_dict("records")}
        print(f"=== {etiqueta} (n={len(sub)}) ===")
        print(ranking.head(10).to_string(index=False))
        print()

    # Correlacion de Spearman entre ambos rankings de importancia (sobre variables comunes)
    m = rankings["hombres"].set_index("variable")["shap_medio_abs"]
    w = rankings["mujeres"].set_index("variable")["shap_medio_abs"]
    comunes = m.index.intersection(w.index)
    rho = pd.Series(m[comunes]).corr(pd.Series(w[comunes]), method="spearman")
    print(f"Correlacion de Spearman entre los rankings de importancia hombres/mujeres: {rho:.3f}")
    resultados["correlacion_spearman_rankings"] = float(rho)

    with open(OUT_DIR / "shap_por_sexo.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nOK ->", OUT_DIR / "shap_por_sexo.json")


if __name__ == "__main__":
    main()
