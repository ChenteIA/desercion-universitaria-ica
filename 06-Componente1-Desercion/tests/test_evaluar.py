import numpy as np
from evaluar import calcular_metricas, bootstrap_ci
from sklearn.metrics import recall_score


def test_calcular_metricas_devuelve_claves_esperadas():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.1, 0.9, 0.4, 0.2, 0.8]
    resultado = calcular_metricas(y_true, y_pred, y_proba)
    assert set(resultado.keys()) == {"recall", "precision", "f1", "auc_roc", "auc_pr"}


def test_calcular_metricas_recall_correcto():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.4, 0.2]
    resultado = calcular_metricas(y_true, y_pred, y_proba)
    assert resultado["recall"] == 0.5


def test_bootstrap_ci_intervalo_valido():
    np.random.seed(0)
    y_true = np.array([0, 1] * 50)
    y_pred = np.array([0, 1] * 50)
    inferior, superior = bootstrap_ci(y_true, y_pred, recall_score, n_iter=200, random_state=42)
    assert inferior <= 1.0 <= superior
    assert inferior <= superior


def test_bootstrap_ci_es_reproducible():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    r1 = bootstrap_ci(y_true, y_pred, recall_score, n_iter=100, random_state=42)
    r2 = bootstrap_ci(y_true, y_pred, recall_score, n_iter=100, random_state=42)
    assert r1 == r2


from evaluar import bootstrap_diferencia_auc, prueba_mcnemar


def test_bootstrap_diferencia_auc_ci_valido():
    np.random.seed(0)
    y_true = np.array(([0, 1] * 50))
    proba_a = np.array(([0.1, 0.9] * 50))
    proba_b = np.array(([0.4, 0.6] * 50))
    inferior, superior = bootstrap_diferencia_auc(y_true, proba_a, proba_b,
                                                     n_iter=200, random_state=42)
    assert inferior <= superior


def test_prueba_mcnemar_devuelve_pvalor():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    pred_a = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    pred_b = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    resultado = prueba_mcnemar(y_true, pred_a, pred_b)
    assert "estadistico" in resultado and "p_valor" in resultado
    assert 0.0 <= resultado["p_valor"] <= 1.0


import pandas as pd
from evaluar import metricas_por_subgrupo


def test_metricas_por_subgrupo_una_fila_por_grupo():
    df = pd.DataFrame({
        "grupo": ["hombre", "hombre", "mujer", "mujer"],
        "y_true": [0, 1, 0, 1],
        "y_pred": [0, 1, 1, 1],
        "y_proba": [0.2, 0.8, 0.6, 0.9],
    })
    resultado = metricas_por_subgrupo(df, "grupo")
    assert set(resultado["grupo"]) == {"hombre", "mujer"}
    assert "recall" in resultado.columns
