import numpy as np
import pandas as pd
from entrenar import separar_ica, construir_pipeline, calibrar_umbral
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def test_separar_ica_excluye_ica_del_nacional():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "target": [0, 1, 0, 1],
        "es_ica": [False, False, True, True],
    })
    nacional, ica = separar_ica(df)
    assert nacional["es_ica"].sum() == 0
    assert len(ica) == 2


def test_separar_ica_preserva_total_de_filas():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
        "es_ica": [False, False, False, True, True],
    })
    nacional, ica = separar_ica(df)
    assert len(nacional) + len(ica) == len(df)


def test_construir_pipeline_imputer_se_ajusta_solo_con_train():
    variables = ["a", "b"]
    df_train_1 = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]})
    df_train_2 = pd.DataFrame({"a": [100.0, 200.0, np.nan], "b": [10.0, 20.0, 30.0]})

    pipe1 = construir_pipeline(RandomForestClassifier(random_state=42), variables, [])
    pipe1.fit(df_train_1, [0, 1, 0])
    media_1 = pipe1.named_steps["preprocesamiento"].transformers_[0][1] \
        .named_steps["imputer"].statistics_[0]

    pipe2 = construir_pipeline(RandomForestClassifier(random_state=42), variables, [])
    pipe2.fit(df_train_2, [0, 1, 0])
    media_2 = pipe2.named_steps["preprocesamiento"].transformers_[0][1] \
        .named_steps["imputer"].statistics_[0]

    assert media_1 != media_2


def test_split_estratificado_preserva_proporcion_de_clase():
    df = pd.DataFrame({
        "x": range(100),
        "target": [0] * 80 + [1] * 20,
    })
    X_train, X_test, y_train, y_test = train_test_split(
        df[["x"]], df["target"], test_size=0.2, stratify=df["target"], random_state=42)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_calibrar_umbral_retorna_valor_en_rango():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])
    umbral = calibrar_umbral(y_true, y_proba)
    assert 0.0 < umbral < 1.0


def test_calibrar_umbral_es_reproducible():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.6, 0.3, 0.7, 0.2, 0.8, 0.55, 0.4, 0.15])
    u1 = calibrar_umbral(y_true, y_proba)
    u2 = calibrar_umbral(y_true, y_proba)
    assert u1 == u2


def test_calibrar_umbral_beta_alto_no_sube_el_umbral():
    # Con beta más alto se pondera más la sensibilidad: el umbral resultante
    # nunca debería ser MÁS ALTO que con un beta menor (beta=1, F1 estándar),
    # en un caso donde subir el umbral reduce sensibilidad.
    # Verificado a mano: en c=0.5 el F1 alcanza su máximo (0.8889) y el
    # F-beta2 también alcanza su máximo (0.9524) en ese mismo punto, por lo
    # que ambos umbrales calibrados coinciden en 0.5 (0.5 <= 0.5 se cumple).
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.2, 0.35, 0.45, 0.55, 0.5, 0.65, 0.75, 0.9])
    umbral_f1 = calibrar_umbral(y_true, y_proba, beta=1.0)
    umbral_f2 = calibrar_umbral(y_true, y_proba, beta=2.0)
    assert umbral_f2 <= umbral_f1
