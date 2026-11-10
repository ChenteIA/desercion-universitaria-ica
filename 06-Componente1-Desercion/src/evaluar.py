import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (recall_score, precision_score, f1_score,
                               roc_auc_score, average_precision_score, confusion_matrix)

from entrenar import CATEGORICAS

BASE_DIR = Path(__file__).parent.parent
MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
TABLAS_DIR = BASE_DIR.parent / "08-Resultados" / "Tablas" / "productivo"
FIGURAS_DIR = BASE_DIR.parent / "08-Resultados" / "Figuras" / "productivo"
UMBRALES_PATH = MODELOS_DIR / "umbrales.json"

RANDOM_STATE = 42


def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
    }


def bootstrap_ci(y_true, y_pred, metrica_fn, n_iter: int = 1000,
                   random_state: int = 42) -> tuple:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    valores = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        valores.append(metrica_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def evaluar() -> dict:
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]

    ica = df[df["es_ica"]]
    X_ica, y_ica = ica[variables].copy(), ica["target"]
    num_vars = [v for v in variables if v not in CATEGORICAS]
    for col in num_vars:
        X_ica[col] = pd.to_numeric(X_ica[col], errors="coerce")
    X_test_nac, y_test_nac = joblib.load(MODELOS_DIR / "test_nacional.pkl")
    with open(UMBRALES_PATH, encoding="utf-8") as f:
        umbrales = json.load(f)

    resultados = {}
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    for nombre in ["rf", "lr", "xgb"]:
        pipeline = joblib.load(MODELOS_DIR / f"pipeline_{nombre}.pkl")

        proba_nac = pipeline.predict_proba(X_test_nac)[:, 1]
        pred_nac = (proba_nac >= umbrales[nombre]).astype(int)
        metricas_nac = calcular_metricas(y_test_nac, pred_nac, proba_nac)

        proba_ica = pipeline.predict_proba(X_ica)[:, 1]
        pred_ica = (proba_ica >= umbrales[nombre]).astype(int)
        metricas_ica = calcular_metricas(y_ica, pred_ica, proba_ica)
        ci_recall_ica = bootstrap_ci(y_ica.values, pred_ica, recall_score)
        ci_auc_ica = bootstrap_ci(y_ica.values, proba_ica, roc_auc_score)

        resultados[nombre] = {
            "nacional": metricas_nac,
            "ica": metricas_ica,
            "ica_ci_recall": ci_recall_ica,
            "ica_ci_auc_roc": ci_auc_ica,
            "umbral": umbrales[nombre],
        }

        matriz = confusion_matrix(y_ica, pred_ica)
        pd.DataFrame(matriz, index=["real_0", "real_1"],
                     columns=["pred_0", "pred_1"]).to_csv(
            TABLAS_DIR / f"matriz_confusion_ica_{nombre}.csv")

    pd.DataFrame(resultados).T.to_csv(TABLAS_DIR / "comparacion_modelos.csv")

    plt.figure(figsize=(6, 4))
    for nombre in ["rf", "lr", "xgb"]:
        plt.bar(nombre, resultados[nombre]["ica"]["auc_roc"])
    plt.axhline(0.80, color="red", linestyle="--", label="criterio Hi (0.80)")
    plt.ylabel("AUC-ROC (Ica)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "comparacion_auc_ica.png", dpi=150)
    plt.close()

    ganador = max(resultados, key=lambda k: resultados[k]["ica"]["auc_roc"])
    otro = sorted((n for n in resultados if n != ganador),
                  key=lambda k: resultados[k]["ica"]["auc_roc"], reverse=True)[0]
    pipeline_ganador = joblib.load(MODELOS_DIR / f"pipeline_{ganador}.pkl")
    pipeline_otro = joblib.load(MODELOS_DIR / f"pipeline_{otro}.pkl")

    proba_ganador = pipeline_ganador.predict_proba(X_ica)[:, 1]
    proba_otro = pipeline_otro.predict_proba(X_ica)[:, 1]
    pred_ganador = (proba_ganador >= umbrales[ganador]).astype(int)
    pred_otro = (proba_otro >= umbrales[otro]).astype(int)

    ci_diferencia_auc = bootstrap_diferencia_auc(y_ica.values, proba_ganador, proba_otro)
    mcnemar_resultado = prueba_mcnemar(y_ica.values, pred_ganador, pred_otro)
    resultados["comparacion_H2"] = {
        "ganador": ganador, "comparado_contra": otro,
        "ic_diferencia_auc": ci_diferencia_auc,
        "mcnemar": mcnemar_resultado,
    }

    df_equidad_sexo = pd.DataFrame({
        "grupo": df.loc[ica.index, "P207"].values,
        "y_true": y_ica.values, "y_pred": pred_ganador, "y_proba": proba_ganador,
    })
    tabla_equidad = metricas_por_subgrupo(df_equidad_sexo, "grupo")
    tabla_equidad.to_csv(TABLAS_DIR / "equidad_por_sexo.csv", index=False)
    resultados["equidad_por_sexo"] = tabla_equidad.to_dict(orient="records")

    df_equidad_area = pd.DataFrame({
        "grupo": df.loc[ica.index, "area_urbana"].map({True: "urbano", False: "rural"}).values,
        "y_true": y_ica.values, "y_pred": pred_ganador, "y_proba": proba_ganador,
    })
    tabla_equidad_area = metricas_por_subgrupo(df_equidad_area, "grupo")
    tabla_equidad_area.to_csv(TABLAS_DIR / "equidad_por_area.csv", index=False)
    resultados["equidad_por_area"] = tabla_equidad_area.to_dict(orient="records")

    with open(TABLAS_DIR / "comparacion_modelos.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(resultados, indent=2, default=str))
    return resultados


from statsmodels.stats.contingency_tables import mcnemar


def bootstrap_diferencia_auc(y_true, proba_a, proba_b, n_iter: int = 1000,
                               random_state: int = 42) -> tuple:
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diferencias = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], proba_a[idx])
        auc_b = roc_auc_score(y_true[idx], proba_b[idx])
        diferencias.append(auc_a - auc_b)
    return float(np.percentile(diferencias, 2.5)), float(np.percentile(diferencias, 97.5))


def prueba_mcnemar(y_true, pred_a, pred_b) -> dict:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correcto_a = (pred_a == y_true)
    correcto_b = (pred_b == y_true)
    tabla = np.array([
        [np.sum(correcto_a & correcto_b), np.sum(correcto_a & ~correcto_b)],
        [np.sum(~correcto_a & correcto_b), np.sum(~correcto_a & ~correcto_b)],
    ])
    resultado = mcnemar(tabla, exact=(tabla[0, 1] + tabla[1, 0] < 25))
    return {"estadistico": float(resultado.statistic), "p_valor": float(resultado.pvalue)}


def metricas_por_subgrupo(df: pd.DataFrame, col_grupo: str) -> pd.DataFrame:
    filas = []
    for grupo, sub in df.groupby(col_grupo):
        m = calcular_metricas(sub["y_true"], sub["y_pred"], sub["y_proba"])
        m["grupo"] = grupo
        m["n"] = len(sub)
        filas.append(m)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    evaluar()
