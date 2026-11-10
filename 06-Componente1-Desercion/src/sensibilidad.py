import json
import joblib
import pandas as pd
from pathlib import Path

from consolidar import consolidar, OUTPUT_PATH_SENSIBILIDAD
from entrenar import entrenar, MODELOS_DIR, VARIABLES_PATH, CATEGORICAS
from evaluar import calcular_metricas, bootstrap_ci
from sklearn.metrics import recall_score, roc_auc_score

AUC_ROC_MINIMO = 0.80
RECALL_MINIMO = 0.75

TABLAS_DIR = Path(__file__).parent.parent.parent / "08-Resultados" / "Tablas" / "productivo"
UMBRALES_SENSIBILIDAD_PATH = MODELOS_DIR / "sensibilidad_umbrales.json"


def comparar_definiciones(metricas_original: dict, metricas_sensibilidad: dict) -> dict:
    original_ok = (metricas_original["auc_roc"] >= AUC_ROC_MINIMO
                   and metricas_original["recall"] >= RECALL_MINIMO)
    sensibilidad_ok = (metricas_sensibilidad["auc_roc"] >= AUC_ROC_MINIMO
                        and metricas_sensibilidad["recall"] >= RECALL_MINIMO)
    return {
        "original_cumple_criterios": original_ok,
        "sensibilidad_cumple_criterios": sensibilidad_ok,
        "conclusion_estable": original_ok == sensibilidad_ok,
        "diferencia_auc_roc": metricas_sensibilidad["auc_roc"] - metricas_original["auc_roc"],
        "diferencia_recall": metricas_sensibilidad["recall"] - metricas_original["recall"],
    }


def ejecutar_sensibilidad() -> dict:
    consolidar(incluir_egresados=True)

    entrenar(dataset_path=OUTPUT_PATH_SENSIBILIDAD, prefijo="sensibilidad_")

    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]

    df_sens = pd.read_csv(OUTPUT_PATH_SENSIBILIDAD)
    df_sens["es_ica"] = df_sens["es_ica"].astype(bool)
    ica_sens = df_sens[df_sens["es_ica"]]
    X_ica_sens, y_ica_sens = ica_sens[variables].copy(), ica_sens["target"]
    # Igual que en entrenar(): algunas columnas numéricas del INEI (p. ej.
    # P114B1) llegan como texto con espacios en blanco representando valores
    # faltantes en vez de NaN; se coacciona a numérico para que el imputer
    # de mediana del pipeline ya entrenado pueda operar sobre estos datos
    # de predicción exactamente como lo hizo sobre los de entrenamiento.
    num_vars = [v for v in variables if v not in CATEGORICAS]
    for col in num_vars:
        X_ica_sens[col] = pd.to_numeric(X_ica_sens[col], errors="coerce")

    pipeline_sens = joblib.load(MODELOS_DIR / "pipeline_sensibilidad_rf.pkl")
    with open(UMBRALES_SENSIBILIDAD_PATH, encoding="utf-8") as f:
        umbrales_sens = json.load(f)
    proba_sens = pipeline_sens.predict_proba(X_ica_sens)[:, 1]
    pred_sens = (proba_sens >= umbrales_sens["rf"]).astype(int)
    metricas_sensibilidad = calcular_metricas(y_ica_sens, pred_sens, proba_sens)

    with open(TABLAS_DIR / "comparacion_modelos.json", encoding="utf-8") as f:
        resultados_original = json.load(f)
    metricas_original = resultados_original["rf"]["ica"]

    comparacion = comparar_definiciones(metricas_original, metricas_sensibilidad)
    comparacion["metricas_original"] = metricas_original
    comparacion["metricas_sensibilidad"] = metricas_sensibilidad

    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLAS_DIR / "analisis_sensibilidad.json", "w", encoding="utf-8") as f:
        json.dump(comparacion, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(comparacion, indent=2, default=str))
    return comparacion


if __name__ == "__main__":
    ejecutar_sensibilidad()
