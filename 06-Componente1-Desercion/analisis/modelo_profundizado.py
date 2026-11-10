# -*- coding: utf-8 -*-
"""Analisis complementarios del modelo ganador (RF) para el Informe Final:
   1) curva de calibracion de probabilidades (test nacional + Ica)
   2) desempeno por subgrupos etarios en Ica (17-21 vs 22-29)
   3) contribuciones SHAP individuales de los 3 casos de mayor riesgo en Ica
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, recall_score, precision_score, f1_score

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))
from entrenar import CATEGORICAS  # noqa: E402

MODELOS = BASE / "modelos"
DATASET = BASE.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
VARIABLES = BASE.parent / "05-Datos" / "procesados" / "variables_seleccionadas.json"
FIGURAS = BASE.parent / "08-Resultados" / "Figuras" / "productivo"

df = pd.read_csv(DATASET)
df["es_ica"] = df["es_ica"].astype(bool)
with open(VARIABLES, encoding="utf-8") as f:
    variables = json.load(f)["seleccionadas_finales"]
with open(MODELOS / "umbrales.json", encoding="utf-8") as f:
    umbral_rf = json.load(f)["rf"]

pipe = joblib.load(MODELOS / "pipeline_rf.pkl")
num_vars = [v for v in variables if v not in CATEGORICAS]

ica = df[df["es_ica"]]
X_ica = ica[variables].copy()
for col in num_vars:
    X_ica[col] = pd.to_numeric(X_ica[col], errors="coerce")
y_ica = ica["target"].values
proba_ica = pipe.predict_proba(X_ica)[:, 1]

X_test, y_test = joblib.load(MODELOS / "test_nacional.pkl")
proba_test = pipe.predict_proba(X_test)[:, 1]

# ---------- 1) curva de calibracion ----------
fig, ax = plt.subplots(figsize=(6, 5))
for (yt, pr, nombre, color, ls, bins) in [
        (y_test, proba_test, "Prueba nacional", "#1f77b4", "-", 8),
        (y_ica, proba_ica, "Ica (validación externa)", "#ff7f0e", "--", 5)]:
    frac_pos, media_pred = calibration_curve(yt, pr, n_bins=bins, strategy="quantile")
    brier = brier_score_loss(yt, pr)
    ax.plot(media_pred, frac_pos, marker="o", color=color, linestyle=ls, linewidth=2,
            markersize=5, label=f"{nombre} (Brier = {brier:.3f})")
ax.plot([0, 1], [0, 1], color="#999999", linewidth=1, linestyle=":",
        label="Calibración perfecta")
ax.set_xlabel("Probabilidad media predicha")
ax.set_ylabel("Fracción observada de desertores")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURAS / "calibracion_rf.png", dpi=150)
plt.close()
print("OK calibracion_rf.png")

# ---------- 2) subgrupos etarios en Ica ----------
edad = pd.to_numeric(ica["P208A"]).values
print("\n--- subgrupos etarios (RF, Ica) ---")
for etiqueta, mask in [("17-21", edad <= 21), ("22-29", edad >= 22)]:
    yt, pr = y_ica[mask], proba_ica[mask]
    for umbral, tipo in [(0.5, "0.5"), (umbral_rf, "calibrado")]:
        pred = (pr >= umbral).astype(int)
        print(f"{etiqueta} n={mask.sum()} pos={yt.sum()} umbral={tipo}: "
              f"recall={recall_score(yt, pred):.3f} "
              f"prec={precision_score(yt, pred, zero_division=0):.3f} "
              f"f1={f1_score(yt, pred, zero_division=0):.3f} "
              f"auc={roc_auc_score(yt, pr):.3f}")

# ---------- 3) contribuciones SHAP de los 3 casos de mayor riesgo ----------
ETIQUETAS = {
    "P208A": "Edad", "ingreso_percapita": "Ingreso per cápita",
    "MIEPERHO": "Miembros del hogar", "educacion_jefe_hogar": "Educación jefe hogar",
    "POBREZA": "Pobreza del hogar", "ESTRSOCIAL": "Estrato socioeconómico",
    "P114B1": "Internet en el hogar", "NBI1": "NBI1 vivienda", "NBI2": "NBI2 saneamiento",
    "NBI3": "NBI3 hacinamiento", "NBI4": "NBI4 dependencia", "NBI5": "NBI5 inasistencia",
}
P209_VAL = {"1": "conviviente", "2": "casado(a)", "3": "viudo(a)", "4": "divorciado(a)",
            "5": "separado(a)", "6": "soltero(a)"}
P203_VAL = {"1": "jefe/jefa hogar", "2": "esposo(a)", "3": "hijo(a)"}
P207_VAL = {"1": "hombre", "2": "mujer"}
P301D_VAL = {"1": "estatal", "2": "no estatal"}

def etiqueta_feature(raw):
    name = raw.replace("num__", "").replace("cat__", "")
    if "_" in name:
        base, _, val = name.rpartition("_")
        val = val.rstrip(".0") or val
        if base == "P209":
            return f"Estado civil: {P209_VAL.get(val, val)}"
        if base == "P203":
            return f"Parentesco: {P203_VAL.get(val, val)}"
        if base == "P207":
            return f"Sexo: {P207_VAL.get(val, val)}"
        if base == "P301D":
            return f"Centro de estudios: {P301D_VAL.get(val, val)}"
        if base == "ESTRSOCIAL":
            return f"Estrato socioeconómico: {val}"
        if base == "categoria_ocupacional_jefe":
            return f"Cat. ocupacional jefe: {val}"
        if base == "ocupacion_jefe_hogar":
            return f"Ocupación jefe (CIUO {val})"
        if base == "P300A":
            return f"Lengua materna: {'castellano' if val == '4' else val}"
    return ETIQUETAS.get(name, name)

modelo_rf = pipe.named_steps["modelo"]
preproc = pipe.named_steps["preprocesamiento"]
X_trans = preproc.transform(X_ica)
nombres = preproc.get_feature_names_out()

explainer = shap.TreeExplainer(modelo_rf)
sv = explainer.shap_values(X_trans)
if isinstance(sv, list):
    sv = sv[1]
elif sv.ndim == 3:
    sv = sv[:, :, 1]

VALORES_NUM = {
    "P208A": lambda v: f"{v:.0f} años",
    "ingreso_percapita": lambda v: f"S/ {v:,.0f}",
    "MIEPERHO": lambda v: f"{v:.0f}",
    "P114B1": lambda v: "sí" if v >= 0.5 else "no",
    "POBREZA": lambda v: {1: "pobre extremo", 2: "pobre", 3: "no pobre"}.get(int(v), f"{v:.0f}"),
    "educacion_jefe_hogar": lambda v: f"nivel {v:.0f}",
}

def etiqueta_con_valor(raw, idx_caso, j):
    base_label = etiqueta_feature(raw)
    if raw.startswith("cat__"):
        presente = X_trans[idx_caso, j] >= 0.5
        return f"{base_label} ({'sí' if presente else 'no'})"
    name = raw.replace("num__", "")
    val = pd.to_numeric(X_ica.iloc[idx_caso][name], errors="coerce")
    if name in VALORES_NUM and pd.notna(val):
        return f"{base_label}: {VALORES_NUM[name](val)}"
    return base_label

top3 = np.argsort(proba_ica)[::-1][:3]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=True)
for panel, (ax, idx) in enumerate(zip(axes, top3), start=1):
    contrib = sv[idx]
    orden = np.argsort(np.abs(contrib))[::-1][:9]
    vals = contrib[orden][::-1]
    labs = [etiqueta_con_valor(nombres[j], idx, j) for j in orden][::-1]
    colores = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
    ax.barh(range(len(vals)), vals, color=colores, height=0.65)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Caso {panel} (probabilidad = {proba_ica[idx]:.2f})", fontsize=9)
    ax.grid(alpha=0.2, axis="x")
axes[1].set_xlabel("Contribución SHAP al riesgo estimado")
plt.tight_layout()
plt.savefig(FIGURAS / "shap_casos_individuales.png", dpi=150)
plt.close()
print("OK shap_casos_individuales.png")
