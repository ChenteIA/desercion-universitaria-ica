"""Intervalos de confianza bootstrap del AUC-ROC por subgrupo en la validacion
de Ica (Tablas 16-18) y de la diferencia de AUC entre subgrupos.

Reutiliza el pipeline_rf.pkl productivo (no lo reentrena). Mismo esquema que
src/evaluar.py: 1,000 iteraciones, semilla 42, percentiles 2.5 y 97.5. Para la
diferencia entre dos subgrupos (muestras independientes) se remuestrea cada
subgrupo por separado en cada iteracion. Se anade una prueba de permutacion de
las etiquetas de grupo como contraste complementario de la diferencia.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from entrenar import CATEGORICAS

MODELOS_DIR = BASE_DIR / "modelos"
DATASET_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES_PATH = BASE_DIR.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
OUT_DIR = BASE_DIR / "analisis" / "salidas"
OUT_DIR.mkdir(exist_ok=True)

N_ITER = 1000
SEMILLA = 42


def ic_auc(y, p, rng):
    n, valores = len(y), []
    for _ in range(N_ITER):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        valores.append(roc_auc_score(y[idx], p[idx]))
    return valores


def comparar(y, p, mascara_a, etiqueta_a, etiqueta_b):
    """AUC e IC de cada subgrupo, IC de la diferencia (a - b) y permutacion."""
    ya, pa, yb, pb = y[mascara_a], p[mascara_a], y[~mascara_a], p[~mascara_a]
    auc_a, auc_b = roc_auc_score(ya, pa), roc_auc_score(yb, pb)

    rng = np.random.RandomState(SEMILLA)
    boot_a = ic_auc(ya, pa, rng)
    boot_b = ic_auc(yb, pb, rng)

    rng = np.random.RandomState(SEMILLA)
    difs = []
    for _ in range(N_ITER):
        ia = rng.randint(0, len(ya), len(ya))
        ib = rng.randint(0, len(yb), len(yb))
        if len(np.unique(ya[ia])) < 2 or len(np.unique(yb[ib])) < 2:
            continue
        difs.append(roc_auc_score(ya[ia], pa[ia]) - roc_auc_score(yb[ib], pb[ib]))

    rng = np.random.RandomState(SEMILLA)
    observada, extremos, validas = auc_a - auc_b, 0, 0
    for _ in range(N_ITER):
        m = rng.permutation(mascara_a)
        if len(np.unique(y[m])) < 2 or len(np.unique(y[~m])) < 2:
            continue
        validas += 1
        d = roc_auc_score(y[m], p[m]) - roc_auc_score(y[~m], p[~m])
        extremos += abs(d) >= abs(observada)

    def resumen(etq, yy, auc, boot):
        return {"grupo": etq, "n": int(len(yy)), "positivos": int(yy.sum()), "auc": float(auc),
                "ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "iter_validas": len(boot)}

    # Sensibilidad bajo el umbral de referencia (0.5) y prueba exacta de Fisher
    da, db = int(((pa >= 0.5) & (ya == 1)).sum()), int(((pb >= 0.5) & (yb == 1)).sum())
    na, nb = int(ya.sum()), int(yb.sum())
    p_fisher = fisher_exact([[da, na - da], [db, nb - db]])[1]

    return {
        "sensibilidad_umbral_0.5": {etiqueta_a: f"{da}/{na}", etiqueta_b: f"{db}/{nb}",
                                    "p_fisher": float(p_fisher)},
        "a": resumen(etiqueta_a, ya, auc_a, boot_a),
        "b": resumen(etiqueta_b, yb, auc_b, boot_b),
        "diferencia": float(observada),
        "ic95_diferencia": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
        "iter_validas_diferencia": len(difs),
        "p_permutacion": float((extremos + 1) / (validas + 1)),
    }


def main():
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)
    with open(VARIABLES_PATH, encoding="utf-8") as f:
        variables = json.load(f)["seleccionadas_finales"]
    ica = df[df["es_ica"]]
    X = ica[variables].copy()
    for col in [v for v in variables if v not in CATEGORICAS]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = ica["target"].values.astype(int)
    p = joblib.load(MODELOS_DIR / "pipeline_rf.pkl").predict_proba(X)[:, 1]

    sexo = ica["P207"]
    print("Valores de P207:", sexo.value_counts().to_dict())
    primer_sexo = sorted(sexo.unique())[0]
    edad = pd.to_numeric(ica["P208A"]).values

    salida = {
        "n_iter": N_ITER, "semilla": SEMILLA,
        "auc_global": float(roc_auc_score(y, p)),
        "sexo": comparar(y, p, (sexo == primer_sexo).values, str(primer_sexo),
                         str(sorted(sexo.unique())[1])),
        "edad": comparar(y, p, edad >= 22, "22-29", "17-21"),
        "area": comparar(y, p, ica["area_urbana"].astype(bool).values, "urbano", "rural"),
    }
    print(json.dumps(salida, indent=2, ensure_ascii=False))
    with open(OUT_DIR / "ic_auc_subgrupos.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=2)
    print("\nOK ->", OUT_DIR / "ic_auc_subgrupos.json")


if __name__ == "__main__":
    main()
