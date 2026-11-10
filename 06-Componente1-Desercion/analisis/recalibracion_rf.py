"""Recalibracion de las probabilidades del Random Forest productivo.

Los calibradores (Platt e isotonico) se ajustan SOLO con predicciones fuera de
muestra del conjunto de entrenamiento nacional (mismo esquema de 5 particiones
y semilla que src/entrenar.py uso para calibrar el umbral); ni la prueba
nacional ni Ica intervienen en su ajuste. Se reporta el puntaje de Brier, la
probabilidad media predicha y el AUC-ROC antes y despues, y se guarda la
figura comparativa de curvas de calibracion.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from entrenar import CATEGORICAS, RANDOM_STATE, separar_ica

MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
FIGURAS_DIR = BASE_DIR.parent / "08-Resultados" / "Figuras" / "productivo"
OUT_DIR = BASE_DIR / "analisis" / "salidas"
OUT_DIR.mkdir(exist_ok=True)


def ic_brier(y, p, n_iter=1000, semilla=42):
    rng = np.random.RandomState(semilla)
    n, v = len(y), []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        v.append(brier_score_loss(y[idx], p[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    df = pd.read_csv(DATASET_PATH)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]
    with open(MODELOS_DIR / "umbrales.json", encoding="utf-8") as f:
        umbral = json.load(f)["rf"]
    num_vars = [v for v in variables if v not in CATEGORICAS]
    nacional, ica = separar_ica(df)

    def preparar(d):
        X = d[variables].copy()
        for col in num_vars:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    # Misma particion que src/entrenar.py
    X, y = preparar(nacional), nacional["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_test_guardado, y_test_guardado = joblib.load(MODELOS_DIR / "test_nacional.pkl")
    assert list(X_test.index) == list(X_test_guardado.index), "la particion no coincide con la de entrenar.py"
    assert (np.asarray(y_test) == np.asarray(y_test_guardado)).all()

    pipeline = joblib.load(MODELOS_DIR / "pipeline_rf.pkl")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    y_train = np.asarray(y_train).astype(int)

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(proba_oof.reshape(-1, 1), y_train)
    isotonica = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(proba_oof, y_train)
    calibradores = {
        "original": lambda p: p,
        "platt": lambda p: platt.predict_proba(p.reshape(-1, 1))[:, 1],
        "isotonica": lambda p: isotonica.predict(p),
    }

    conjuntos = {
        "prueba_nacional": (np.asarray(y_test).astype(int), pipeline.predict_proba(X_test)[:, 1]),
        "ica": (ica["target"].values.astype(int), pipeline.predict_proba(preparar(ica))[:, 1]),
    }
    salida = {
        "n_entrenamiento_oof": int(len(y_train)),
        "brier_oof_original": float(brier_score_loss(y_train, proba_oof)),
        "platt": {"pendiente": float(platt.coef_[0, 0]), "intercepto": float(platt.intercept_[0]),
                  "umbral_calibrado_equivalente": float(calibradores["platt"](np.array([umbral]))[0])},
        "umbral_original": umbral,
    }
    for nombre, (yt, p) in conjuntos.items():
        salida[nombre] = {"n": int(len(yt)), "prevalencia": float(yt.mean()),
                          "brier_referencia_prevalencia": float(brier_score_loss(yt, np.full(len(yt), yt.mean())))}
        for metodo, f in calibradores.items():
            pc = f(p)
            salida[nombre][metodo] = {
                "brier": float(brier_score_loss(yt, pc)), "ic95_brier": ic_brier(yt, pc),
                "proba_media": float(pc.mean()), "auc_roc": float(roc_auc_score(yt, pc)),
                "marcados_umbral_0.5": int((pc >= 0.5).sum()),
            }
    print(json.dumps(salida, indent=2, ensure_ascii=False))
    with open(OUT_DIR / "recalibracion_rf.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=2)

    # Figura: curvas de calibracion antes y despues (Platt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    titulos = {"prueba_nacional": "Prueba nacional (n = 667)", "ica": "Ica, validación externa (n = 190)"}
    estilos = [("original", "Original", "#c0392b", "-"), ("platt", "Recalibrado (Platt)", "#1f77b4", "--")]
    for ax, (nombre, (yt, p)) in zip(axes, conjuntos.items()):
        bins = 8 if nombre == "prueba_nacional" else 5
        for metodo, etiqueta, color, ls in estilos:
            pc = calibradores[metodo](p)
            frac, media = calibration_curve(yt, pc, n_bins=bins, strategy="quantile")
            ax.plot(media, frac, marker="o", color=color, linestyle=ls, linewidth=2, markersize=5,
                    label=f"{etiqueta} (Brier = {brier_score_loss(yt, pc):.3f})")
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle=":", label="Calibración perfecta")
        ax.set_title(titulos[nombre], fontsize=10)
        ax.set_xlabel("Probabilidad media predicha")
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Fracción observada de desertores")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "calibracion_rf_recalibrada.png", dpi=150)
    plt.close()
    print("\nOK ->", OUT_DIR / "recalibracion_rf.json", "y calibracion_rf_recalibrada.png")


if __name__ == "__main__":
    main()
