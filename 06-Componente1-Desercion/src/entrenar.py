import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                       RandomizedSearchCV, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
MODELOS_DIR = BASE_DIR / "modelos"

RANDOM_STATE = 42

CATEGORICAS = ["P207", "P203", "P209", "P301D", "P300A",
               "ocupacion_jefe_hogar", "categoria_ocupacional_jefe", "ESTRSOCIAL"]


def separar_ica(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df["es_ica"] = df["es_ica"].astype(bool)
    nacional = df[~df["es_ica"]].copy()
    ica = df[df["es_ica"]].copy()
    return nacional, ica


def construir_pipeline(modelo, variables_numericas: list, variables_categoricas: list) -> Pipeline:
    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocesamiento = ColumnTransformer([
        ("num", numerico, variables_numericas),
        ("cat", categorico, variables_categoricas),
    ])
    return Pipeline([
        ("preprocesamiento", preprocesamiento),
        ("modelo", modelo),
    ])


def calibrar_umbral(y_true, y_proba, beta: float = 2.0) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    candidatos = np.unique(np.round(y_proba, 3))
    candidatos = candidatos[(candidatos > 0) & (candidatos < 1)]
    if len(candidatos) == 0:
        return 0.5
    puntajes = [fbeta_score(y_true, (y_proba >= c).astype(int), beta=beta, zero_division=0)
                for c in candidatos]
    return float(candidatos[int(np.argmax(puntajes))])


def entrenar(dataset_path: Path = DATASET_PATH, prefijo: str = "") -> dict:
    df = pd.read_csv(dataset_path)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]

    nacional, _ica = separar_ica(df)
    num_vars = [v for v in variables if v not in CATEGORICAS]
    cat_vars = [v for v in variables if v in CATEGORICAS]

    X = nacional[variables].copy()
    # Algunas columnas numéricas del INEI (p. ej. P114B1) llegan como texto
    # con espacios en blanco representando valores faltantes en vez de NaN;
    # se coacciona a numérico para que el imputer de mediana pueda operar.
    for col in num_vars:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = nacional["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    modelos = {
        "rf": (RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
               {"modelo__n_estimators": [200, 300, 500],
                "modelo__max_depth": [None, 10, 20],
                "modelo__min_samples_leaf": [1, 5, 10]}),
        "lr": (LogisticRegression(class_weight="balanced", max_iter=1000,
                                    random_state=RANDOM_STATE),
               {"modelo__C": [0.01, 0.1, 1.0, 10.0]}),
        "xgb": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
                {"modelo__n_estimators": [200, 300], "modelo__max_depth": [3, 5, 7],
                 "modelo__learning_rate": [0.01, 0.1, 0.3]}),
    }

    resumen = {}
    umbrales = {}
    MODELOS_DIR.mkdir(exist_ok=True)
    for nombre, (modelo, param_grid) in modelos.items():
        pipeline = construir_pipeline(modelo, num_vars, cat_vars)
        busqueda = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=cv,
                                        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1)
        busqueda.fit(X_train, y_train)
        joblib.dump(busqueda.best_estimator_, MODELOS_DIR / f"pipeline_{prefijo}{nombre}.pkl")
        resumen[nombre] = {"mejor_score_cv": busqueda.best_score_,
                            "mejores_params": busqueda.best_params_}
        print(nombre, resumen[nombre])

        proba_oof = cross_val_predict(busqueda.best_estimator_, X_train, y_train,
                                        cv=cv, method="predict_proba")[:, 1]
        umbral = calibrar_umbral(y_train.values if hasattr(y_train, "values") else y_train,
                                  proba_oof, beta=2.0)
        umbrales[nombre] = umbral

    joblib.dump((X_test, y_test), MODELOS_DIR / f"{prefijo}test_nacional.pkl")
    with open(MODELOS_DIR / f"{prefijo}umbrales.json", "w", encoding="utf-8") as f:
        json.dump(umbrales, f, ensure_ascii=False, indent=2)
    return resumen


if __name__ == "__main__":
    entrenar()
