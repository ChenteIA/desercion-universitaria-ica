import json
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from entrenar import CATEGORICAS

BASE_DIR = Path(__file__).parent.parent
MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
FIGURAS_DIR = BASE_DIR.parent / "08-Resultados" / "Figuras" / "productivo"

COLUMNAS_IDENTIFICATORIAS = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]


def anonimizar_caso(fila: pd.Series, columnas_prohibidas: list) -> pd.Series:
    return fila.drop(labels=[c for c in columnas_prohibidas if c in fila.index])


def calcular_shap(modelo, X: pd.DataFrame):
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        # shap>=0.45 devuelve (n_muestras, n_features, n_clases) para
        # clasificadores binarios en vez de una lista de arrays por clase;
        # se toma la clase positiva (índice 1).
        shap_values = shap_values[:, :, 1]
    return shap_values


def interpretar() -> None:
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]

    ica = df[df["es_ica"]]
    X_ica = ica[variables].copy()
    num_vars = [v for v in variables if v not in CATEGORICAS]
    for col in num_vars:
        X_ica[col] = pd.to_numeric(X_ica[col], errors="coerce")

    pipeline = joblib.load(MODELOS_DIR / "pipeline_rf.pkl")
    modelo_rf = pipeline.named_steps["modelo"]
    preprocesamiento = pipeline.named_steps["preprocesamiento"]
    X_transformado = preprocesamiento.transform(X_ica)
    nombres_columnas = preprocesamiento.get_feature_names_out()

    shap_values = calcular_shap(modelo_rf, X_transformado)

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_transformado, feature_names=nombres_columnas, show=False)
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "shap_global_ica.png", dpi=150)
    plt.close()

    casos_riesgo = ica.assign(proba=pipeline.predict_proba(X_ica)[:, 1]) \
        .nlargest(3, "proba")
    casos_anonimizados = [
        anonimizar_caso(fila, COLUMNAS_IDENTIFICATORIAS).to_dict()
        for _, fila in casos_riesgo.iterrows()
    ]
    with open(FIGURAS_DIR.parent.parent / "Tablas" / "productivo" / "casos_riesgo_ica.json",
              "w", encoding="utf-8") as f:
        json.dump(casos_anonimizados, f, ensure_ascii=False, indent=2, default=str)

    print(f"SHAP global guardado. {len(casos_anonimizados)} casos ilustrativos anonimizados.")


if __name__ == "__main__":
    interpretar()
