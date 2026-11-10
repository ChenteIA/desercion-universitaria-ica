# -*- coding: utf-8 -*-
"""Comparacion SMOTE vs ponderacion de clases sobre la configuracion definitiva
de 19 predictores, replicando la comparacion de la fase exploratoria (T05).
Mismo split y CV que entrenar.py (semilla 42); mejores hiperparametros del RF
de la Task 3. SMOTE se aplica solo dentro del fold de entrenamiento (imblearn)."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
from entrenar import CATEGORICAS, construir_pipeline, separar_ica  # noqa: E402

DATASET = BASE.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES = BASE.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
RANDOM_STATE = 42
MEJORES_RF = dict(n_estimators=500, min_samples_leaf=10, max_depth=10)

df = pd.read_csv(DATASET)
df["es_ica"] = df["es_ica"].astype(bool)
with open(VARIABLES, encoding="utf-8") as f:
    variables = json.load(f)["seleccionadas_finales"]

nacional, _ = separar_ica(df)
num_vars = [v for v in variables if v not in CATEGORICAS]
cat_vars = [v for v in variables if v in CATEGORICAS]
X = nacional[variables].copy()
for col in num_vars:
    X[col] = pd.to_numeric(X[col], errors="coerce")
y = nacional["target"]
X_train, _X_test, y_train, _y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"recall": "recall", "auc_pr": "average_precision", "auc_roc": "roc_auc"}

pipe_ponderacion = construir_pipeline(
    RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                           n_jobs=-1, **MEJORES_RF),
    num_vars, cat_vars)

preproc = construir_pipeline(RandomForestClassifier(), num_vars, cat_vars) \
    .named_steps["preprocesamiento"]
pipe_smote = ImbPipeline([
    ("preprocesamiento", preproc),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("modelo", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **MEJORES_RF)),
])

for nombre, pipe in [("ponderacion", pipe_ponderacion), ("SMOTE", pipe_smote)]:
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    print(f"{nombre}: recall={res['test_recall'].mean():.3f}  "
          f"auc_pr={res['test_auc_pr'].mean():.3f}  "
          f"auc_roc={res['test_auc_roc'].mean():.3f}")
