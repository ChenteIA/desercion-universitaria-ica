import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from seleccionar import filtrar_completitud, rankear_importancia


def test_filtrar_completitud_descarta_columna_con_muchos_nulos():
    df = pd.DataFrame({
        "buena": [1, 2, 3, 4, 5],
        "mala": [1, None, None, None, None],
    })
    resultado = filtrar_completitud(df, ["buena", "mala"], umbral=0.4)
    assert resultado == ["buena"]


def test_rankear_importancia_nunca_recibe_filas_de_ica():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 1, 2, 3, 4],
        "x2": [5, 6, 7, 8, 5, 6, 7, 8],
        "target": [0, 0, 1, 1, 0, 0, 1, 1],
        "es_ica": [False, False, False, False, True, True, True, True],
    })
    llamadas = {}
    df_original_fit = RandomForestClassifier.fit

    def fit_espia(self, X, y, *args, **kwargs):
        llamadas["n_filas_entrenadas"] = len(X)
        return df_original_fit(self, X, y, *args, **kwargs)

    RandomForestClassifier.fit = fit_espia
    try:
        rankear_importancia(df, ["x1", "x2"])
    finally:
        RandomForestClassifier.fit = df_original_fit

    assert llamadas["n_filas_entrenadas"] == 4  # solo las 4 filas con es_ica==False


def test_rankear_importancia_retorna_orden_descendente():
    df = pd.DataFrame({
        "x1": list(range(20)),
        "x2": [1] * 20,
        "target": [0, 1] * 10,
        "es_ica": [False] * 20,
    })
    resultado = rankear_importancia(df, ["x1", "x2"])
    assert list(resultado["importancia"]) == sorted(resultado["importancia"], reverse=True)
