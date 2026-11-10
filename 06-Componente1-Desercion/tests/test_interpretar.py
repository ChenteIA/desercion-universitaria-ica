import pandas as pd
from interpretar import anonimizar_caso


def test_anonimizar_caso_elimina_columnas_prohibidas():
    fila = pd.Series({"CONGLOME": "001", "VIVIENDA": "01", "CODPERSO": "02", "P208A": 22})
    resultado = anonimizar_caso(fila, columnas_prohibidas=["CONGLOME", "VIVIENDA", "CODPERSO"])
    assert set(resultado.index) == {"P208A"}


def test_anonimizar_caso_conserva_columnas_permitidas():
    fila = pd.Series({"CONGLOME": "001", "P208A": 22, "target": 1})
    resultado = anonimizar_caso(fila, columnas_prohibidas=["CONGLOME"])
    assert resultado["P208A"] == 22
    assert resultado["target"] == 1


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from interpretar import calcular_shap


def test_calcular_shap_forma_correcta():
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = [0, 1] * 10
    modelo = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    valores = calcular_shap(modelo, X)
    assert valores.shape == (20, 2)
