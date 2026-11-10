"""Deciles de riesgo (ganancia/lift) y curvas de dependencia parcial del RF.

Anclas: Ica 190 casos / 42 desertores (22.1%); prueba nacional 667 casos.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

TESIS = Path(__file__).parent.parent.parent
REPO = Path(__file__).parent.parent
CSV = TESIS / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARS = TESIS / "05-Datos" / "procesados" / "variables_seleccionadas.json"
TABLAS = TESIS / "08-Resultados" / "Tablas" / "productivo"
FIGURAS = TESIS / "08-Resultados" / "Figuras" / "productivo"


def cargar():
    pipe = joblib.load(REPO / "modelos" / "pipeline_rf.pkl")
    x_test, y_test = joblib.load(REPO / "modelos" / "test_nacional.pkl")
    df = pd.read_csv(CSV)
    # patron obligatorio ENAHO: P114B1 usa ' ' como nulo
    df["P114B1"] = pd.to_numeric(df["P114B1"], errors="coerce")
    cols = json.load(open(VARS, encoding="utf-8"))["seleccionadas_finales"]
    es_ica = df["es_ica"].astype(str).isin(["True", "1", "1.0"])
    x_ica, y_ica = df.loc[es_ica, cols], df.loc[es_ica, "target"].astype(int)
    x_nac = df.loc[~es_ica, cols]
    assert len(x_ica) == 190 and int(y_ica.sum()) == 42, "Ica no cuadra"
    assert len(x_test) == 667, "prueba nacional no cuadra"
    return pipe, x_test, y_test.astype(int), x_ica, y_ica, x_nac


def deciles(pipe, x, y, nombre):
    proba = pipe.predict_proba(x)[:, 1]
    orden = np.argsort(-proba)
    y_ord = y.to_numpy()[orden]
    grupos = np.array_split(y_ord, 10)
    total_pos, total_n = int(y.sum()), len(y)
    filas, acum_pos, acum_n = [], 0, 0
    for d, g in enumerate(grupos, start=1):
        acum_pos += int(g.sum()); acum_n += len(g)
        filas.append({
            "conjunto": nombre, "decil": d, "n": len(g), "desertores": int(g.sum()),
            "pct_poblacion_acum": 100 * acum_n / total_n,
            "pct_capturado_acum": 100 * acum_pos / total_pos,
            "lift_acum": (acum_pos / total_pos) / (acum_n / total_n),
        })
    return pd.DataFrame(filas)


def figura_ganancia(t):
    fig, ax = plt.subplots(figsize=(6, 4))
    for nombre, color in [("Prueba nacional", "#61a5c2"), ("Validación Ica", "#2a6f97")]:
        d = t[t["conjunto"] == nombre]
        ax.plot([0] + list(d["pct_poblacion_acum"]), [0] + list(d["pct_capturado_acum"]),
                marker="o", ms=3, lw=1.2, color=color, label=nombre)
    ax.plot([0, 100], [0, 100], ls="--", lw=1, color="gray", label="Azar")
    ax.set_xlabel("Población priorizada por puntaje de riesgo (%)", fontsize=9)
    ax.set_ylabel("Desertores capturados (%)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS / "ganancia_acumulada.png", dpi=150)
    plt.close(fig)


def figura_pdp(pipe, x_nac):
    # PDP exige dtypes flotantes en las variables analizadas
    x_nac = x_nac.astype({"ingreso_percapita": float, "P208A": float, "MIEPERHO": float})
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3))
    PartialDependenceDisplay.from_estimator(
        pipe, x_nac, ["ingreso_percapita", "P208A", "MIEPERHO"], ax=axes,
        line_kw={"color": "#2a6f97", "lw": 1.4})
    etiquetas = ["Ingreso per cápita anual del hogar (S/)", "Edad (años)", "Miembros del hogar"]
    for ax, et in zip(axes, etiquetas):
        ax.set_xlabel(et, fontsize=8)
        ax.set_ylabel("Dependencia parcial", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURAS / "pdp_variables.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    pipe, x_test, y_test, x_ica, y_ica, x_nac = cargar()
    t = pd.concat([deciles(pipe, x_test, y_test, "Prueba nacional"),
                   deciles(pipe, x_ica, y_ica, "Validación Ica")], ignore_index=True)
    t.to_csv(TABLAS / "deciles_riesgo.csv", index=False, encoding="utf-8")
    figura_ganancia(t)
    figura_pdp(pipe, x_nac)
    print(t.round(2).to_string(index=False))
    print("OK")
