"""Corrige por comparaciones multiples las 11 pruebas de independencia
chi-cuadrado reportadas en la Tabla 9 del Informe (seccion 3.2).

Los p-valores exactos se reconstruyen analiticamente a partir del
estadistico chi2 y los grados de libertad ya publicados en la Tabla 9
(stats.chi2.sf), sin necesidad de volver a correr las pruebas sobre el
dataset crudo. Se aplican dos correcciones estandar para 11 pruebas
simultaneas a alfa=0.05: Bonferroni (conservadora) y Benjamini-Hochberg
/ FDR (recomendada, menos conservadora).
"""
import json
from pathlib import Path

from scipy import stats

OUT_DIR = Path(__file__).parent / "salidas"
OUT_DIR.mkdir(exist_ok=True)

# (variable, estadistico chi2, grados de libertad) tal como figuran en la Tabla 9
FILAS = [
    ("Edad", 250.8, 4),
    ("Pobreza del hogar", 45.6, 2),
    ("Miembros del hogar", 21.2, 4),
    ("NBI2 (servicios higienicos)", 5.2, 1),
    ("NBI3 (hacinamiento)", 3.8, 1),
    ("NBI1 (vivienda inadecuada)", 3.6, 1),
    ("Tipo de centro de estudios", 3.1, 1),
    ("Area de residencia", 0.8, 1),
    ("Sexo", 0.2, 1),
    ("NBI5 (inasistencia escolar)", 0.0, 1),
    ("NBI4 (dependencia economica)", 0.0, 1),
]


def benjamini_hochberg(pvalores: list) -> list:
    n = len(pvalores)
    orden = sorted(range(n), key=lambda i: pvalores[i])
    ajustados = [0.0] * n
    previo = 1.0
    for rango, idx in enumerate(reversed(orden), start=1):
        k = n - rango + 1
        val = min(previo, pvalores[idx] * n / k)
        ajustados[idx] = val
        previo = val
    return ajustados


def main():
    n = len(FILAS)
    p_exactos = [float(stats.chi2.sf(chi2, gl)) for _, chi2, gl in FILAS]
    bonferroni = [min(1.0, p * n) for p in p_exactos]
    fdr = benjamini_hochberg(p_exactos)

    resultado = []
    for (nombre, chi2, gl), p, pb, pf in zip(FILAS, p_exactos, bonferroni, fdr):
        resultado.append({
            "variable": nombre, "chi2": chi2, "gl": gl,
            "p_exacto": p, "p_bonferroni": pb, "p_fdr": pf,
            "significativa_fdr_0.05": pf < 0.05,
        })
        print(f"{nombre:32s} chi2={chi2:6.1f} p={p:.2e}  Bonferroni={pb:.4f}  "
              f"FDR={pf:.4f}  {'Sig.' if pf < 0.05 else 'No sig.'} (FDR)")

    with open(OUT_DIR / "correccion_multiple_chi2.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print("\nOK ->", OUT_DIR / "correccion_multiple_chi2.json")


if __name__ == "__main__":
    main()
