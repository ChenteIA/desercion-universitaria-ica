"""Verifica si los hiperparametros optimos de RF/LR reportados en la Tabla 2
del Informe (obtenidos con RandomizedSearchCV, n_iter=10) siguen siendo
optimos al ampliar la rejilla mas alla del borde donde cayeron (500 arboles,
10 obs. minimas por hoja para RF; C=0.01 para LR) o al hacer grid search
exhaustivo en vez de aleatorio.

No modifica ningun artefacto productivo (modelos/*.pkl, umbrales.json):
usa exactamente el mismo split/CV que src/entrenar.py (misma semilla, mismos
folds) mediante import directo de sus funciones, y solo imprime/guarda un
resumen de verificacion en analisis/salidas/.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from entrenar import (CATEGORICAS, DATASET_PATH, RANDOM_STATE, VARIABLES_PATH,
                       construir_pipeline, separar_ica)

OUT_DIR = BASE_DIR / "analisis" / "salidas"
OUT_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATASET_PATH)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]

    nacional, _ica = separar_ica(df)
    num_vars = [v for v in variables if v not in CATEGORICAS]
    cat_vars = [v for v in variables if v in CATEGORICAS]

    X = nacional[variables].copy()
    for col in num_vars:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = nacional["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultados = {}

    # ---------- Random Forest: grid original (27 combos) vs extendida ----------
    print("=== Random Forest ===")
    rf_original = {"modelo__n_estimators": [200, 300, 500],
                    "modelo__max_depth": [None, 10, 20],
                    "modelo__min_samples_leaf": [1, 5, 10]}
    rf_extendida = {"modelo__n_estimators": [200, 300, 500, 800, 1000],
                     "modelo__max_depth": [None, 10, 20, 30],
                     "modelo__min_samples_leaf": [1, 5, 10, 15, 20]}

    for etiqueta, grid in [("original_exhaustiva", rf_original), ("extendida_exhaustiva", rf_extendida)]:
        t0 = time.time()
        modelo = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        pipeline = construir_pipeline(modelo, num_vars, cat_vars)
        busqueda = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        busqueda.fit(X_train, y_train)
        dt = time.time() - t0
        print(f"[rf/{etiqueta}] mejor CV AUC-ROC={busqueda.best_score_:.4f} "
              f"params={busqueda.best_params_} ({dt:.0f}s, {len(busqueda.cv_results_['params'])} combos)")
        resultados[f"rf_{etiqueta}"] = {
            "mejor_score_cv": busqueda.best_score_,
            "mejores_params": busqueda.best_params_,
            "n_combinaciones": len(busqueda.cv_results_["params"]),
            "segundos": dt,
        }

    # ---------- Regresion logistica: rejilla extendida hacia mayor regularizacion ----------
    print("=== Regresion logistica ===")
    lr_extendida = {"modelo__C": [0.001, 0.003, 0.01, 0.1, 1.0, 10.0]}
    t0 = time.time()
    modelo = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    pipeline = construir_pipeline(modelo, num_vars, cat_vars)
    busqueda = GridSearchCV(pipeline, lr_extendida, cv=cv, scoring="roc_auc", n_jobs=-1)
    busqueda.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"[lr/extendida] mejor CV AUC-ROC={busqueda.best_score_:.4f} "
          f"params={busqueda.best_params_} ({dt:.0f}s)")
    resultados["lr_extendida"] = {
        "mejor_score_cv": busqueda.best_score_,
        "mejores_params": busqueda.best_params_,
        "n_combinaciones": len(busqueda.cv_results_["params"]),
        "segundos": dt,
    }

    with open(OUT_DIR / "verificacion_rejilla_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nOK ->", OUT_DIR / "verificacion_rejilla_hiperparametros.json")


if __name__ == "__main__":
    main()
