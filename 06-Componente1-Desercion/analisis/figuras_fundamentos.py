"""Figuras ilustrativas de la seccion 2.4 del Informe Final (no usan datos ENAHO).

Genera:
  esquema_random_forest.png   - diagrama bootstrap -> arboles -> votacion
  curvas_roc_ilustrativas.png - ROC de clasificador perfecto, aleatorio y tipico
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import auc, roc_curve

SALIDA = Path(__file__).parent.parent.parent / "08-Resultados" / "Figuras" / "ilustrativas"
SEMILLA = 42


def _caja(ax, x, y, w, h, texto, fc="#f2f2f2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=8)


def _flecha(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=12, lw=0.8, color="black"))


def esquema_random_forest():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _caja(ax, 3.4, 8.6, 3.2, 1.0, "Datos de entrenamiento\n(muestra nacional)")
    columnas = [(0.6, "1"), (3.9, "2"), (7.2, "B")]
    for x, k in columnas:
        _flecha(ax, 5.0, 8.55, x + 1.1, 7.15)
        _caja(ax, x, 6.1, 2.2, 1.0, f"Muestra bootstrap {k}", fc="#e0e8f0")
        _flecha(ax, x + 1.1, 6.05, x + 1.1, 5.15)
        _caja(ax, x, 3.9, 2.2, 1.2, f"Árbol {k}\n(subconjunto aleatorio\nde variables por nodo)")
        _flecha(ax, x + 1.1, 3.85, 5.0, 2.65)
    ax.text(6.65, 6.6, "· · ·", ha="center", fontsize=11)
    ax.text(6.65, 4.5, "· · ·", ha="center", fontsize=11)
    _caja(ax, 3.4, 1.5, 3.2, 1.1, "Agregación:\nvotación / promedio", fc="#e0e8f0")
    _flecha(ax, 5.0, 1.45, 5.0, 0.75)
    ax.text(5.0, 0.35, "Predicción del ensamble", ha="center", fontsize=9,
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(SALIDA / "esquema_random_forest.png", dpi=150)
    plt.close(fig)


def curvas_roc_ilustrativas():
    rng = np.random.default_rng(SEMILLA)
    n = 2000
    y = np.concatenate([np.zeros(n), np.ones(n)])
    # d' = 0.954 -> AUC teorica = Phi(0.954/sqrt(2)) ~ 0.75
    puntajes_tipico = np.concatenate([rng.normal(0.0, 1.0, n),
                                      rng.normal(0.954, 1.0, n)])
    fpr_t, tpr_t, _ = roc_curve(y, puntajes_tipico)
    auc_t = auc(fpr_t, tpr_t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 0, 1], [0, 1, 1], lw=1.2, color="#2a6f97",
            label="Clasificador perfecto (AUC = 1.00)")
    ax.plot(fpr_t, tpr_t, lw=1.2, color="#61a5c2",
            label=f"Clasificador típico (AUC = {auc_t:.2f})")
    ax.plot([0, 1], [0, 1], lw=1.0, ls="--", color="gray",
            label="Clasificador aleatorio (AUC = 0.50)")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)", fontsize=9)
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(SALIDA / "curvas_roc_ilustrativas.png", dpi=150)
    plt.close(fig)


def figura_flujo_pipeline():
    fases = [
        ("Fase 1 — Consolidación de la base de datos",
         "5 módulos ENAHO 2024 → 3,523 registros\n(888 desertores; Ica: 190/42)", "#f2f2f2"),
        ("Fase 2 — Selección de variables",
         "19 predictoras (completitud +\nimportancia Random Forest)", "#e0e8f0"),
        ("Fase 3 — Entrenamiento",
         "RF, regresión logística y XGBoost\nCV estratificada (5) + umbral F-beta (β = 2)", "#e0e8f0"),
        ("Fase 4 — Evaluación",
         "Prueba nacional (20%) y validación externa\nen Ica, con IC bootstrap del 95%", "#f2f2f2"),
        ("Fase 5 — Interpretación",
         "SHAP global e individual (TreeExplainer)", "#f2f2f2"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 7.0))
    ax.set_xlim(0, 10); ax.set_ylim(1.3, 12.2); ax.axis("off")
    y = 12.0
    alto, paso = 1.55, 2.15
    for i, (titulo, detalle, color) in enumerate(fases):
        _caja(ax, 0.5, y - alto, 5.6, alto, f"{titulo}\n{detalle}", fc=color)
        if i < len(fases) - 1:
            _flecha(ax, 3.3, y - alto - 0.05, 3.3, y - paso + 0.05)
        y -= paso
    # exclusion de Ica (banda lateral, fases 2-4)
    ax.add_patch(FancyBboxPatch((6.7, 4.3), 3.0, 4.6, boxstyle="round,pad=0.02",
                                fc="white", ec="#2a6f97", lw=1.2, ls="--"))
    ax.text(8.2, 6.6, "La región Ica queda\nexcluida de todo ajuste\n(fases 2 y 3) y se reserva\n"
                      "para la validación\nexterna (fase 4)",
            ha="center", va="center", fontsize=8, color="#2a6f97")
    # rama del analisis de sensibilidad
    _caja(ax, 6.7, 1.6, 3.0, 1.7,
          "Análisis de sensibilidad:\nclase negativa alternativa\n(réplica de las fases 1 a 4)",
          fc="#f2f2f2")
    _flecha(ax, 6.15, 4.2, 7.7, 3.35)
    fig.tight_layout()
    fig.savefig(SALIDA / "flujo_pipeline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    esquema_random_forest()
    curvas_roc_ilustrativas()
    figura_flujo_pipeline()
    print("OK:", *[p.name for p in sorted(SALIDA.glob("*.png"))])
