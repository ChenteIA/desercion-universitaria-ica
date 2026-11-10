"""Matriz de correlaciones de Spearman entre predictores numericos/ordinales.

Reconstruye la Figura 6 (correlaciones_predictores.png, Capitulo III), que no
tenia script de regeneracion propio en el repo. Usa el dataset consolidado
completo (3,523 filas, incluye Ica) -- a diferencia de la prueba chi-cuadrado
de la seccion 3.2, que excluye Ica; se verifico empiricamente contra los
valores ya publicados en el texto (0.43/-0.37/0.34/0.29) que esa es la
poblacion sobre la que se calculo originalmente esta figura.

Anclas del texto ya publicado (Capitulo-III-Resultados.md, seccion 3.2):
max |rho| = 0.43 entre ingreso_percapita y educacion_jefe_hogar; -0.37 con
ESTRSOCIAL; 0.34 con POBREZA; 0.29 con P114B1 (internet).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TESIS = Path(__file__).parent.parent.parent
CSV = TESIS / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
FIGURAS = TESIS / "08-Resultados" / "Figuras" / "productivo"

# Subconjunto de variables de naturaleza numerica u ordinal entre las 19
# seleccionadas (excluye las estrictamente nominales: sexo, estado civil,
# parentesco, area, tipo de centro, ocupacion/categoria ocupacional del jefe).
COLUMNAS = {
    "P208A": "Edad",
    "ingreso_percapita": "Ingreso per cápita",
    "MIEPERHO": "Miembros del hogar",
    "educacion_jefe_hogar": "Educación jefe hogar",
    "POBREZA": "Pobreza (1-3)",
    "ESTRSOCIAL": "Estrato socioeconómico",
    "P114B1": "Internet en el hogar",
    "NBI1": "NBI1 vivienda",
    "NBI2": "NBI2 saneamiento",
    "NBI3": "NBI3 hacinamiento",
    "NBI4": "NBI4 dependencia econ.",
    "NBI5": "NBI5 inasistencia esc.",
}


def cargar_completo():
    df = pd.read_csv(CSV)
    for col in COLUMNAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    assert len(df) == 3523, "dataset consolidado no cuadra"
    return df[list(COLUMNAS)].rename(columns=COLUMNAS)


def main():
    datos = cargar_completo()
    corr = datos.corr(method="spearman")

    fuera_diagonal = corr.values[~np.eye(len(corr), dtype=bool)]
    maximo = np.max(np.abs(fuera_diagonal))
    # Ancla contra el valor ya publicado en Capitulo-III-Resultados.md (rho=0.43
    # entre ingreso_percapita y educacion_jefe_hogar).
    assert round(maximo, 2) == 0.43, f"correlacion maxima no coincide con lo publicado: {maximo:.3f}"

    etiquetas = corr.columns.tolist()
    n = len(etiquetas)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticklabels(etiquetas)

    for i in range(n):
        for j in range(n):
            valor = corr.values[i, j]
            color = "white" if abs(valor) > 0.6 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación de Spearman")

    plt.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURAS / "correlaciones_predictores.png", dpi=150)
    plt.close()
    print(f"OK correlaciones_predictores.png (max |rho| fuera de diagonal = {maximo:.3f})")


if __name__ == "__main__":
    main()
