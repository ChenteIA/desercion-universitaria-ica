"""Potencia de las pruebas chi-cuadrado de la Tabla 9 si se hubieran aplicado
solo al subconjunto de Ica (n = 190, 42 desertores), en vez de a la poblacion
nacional (n = 3,523). Sustenta la justificacion de la seccion 3.2 y de C1.

El tamano del efecto de cada asociacion se toma de la propia Tabla 9 como
w de Cohen = sqrt(chi2 / n_nacional), suponiendo que la asociacion en Ica es
de la misma magnitud que a nivel nacional. La potencia se calcula con la
distribucion chi-cuadrado no central (no centralidad = n * w^2) a alfa = 0.05.
Tambien se reporta el w minimo detectable con potencia de 0.80 en Ica.
"""
import json
from pathlib import Path

from scipy import optimize, stats

OUT_DIR = Path(__file__).parent / "salidas"
OUT_DIR.mkdir(exist_ok=True)

ALFA = 0.05
POTENCIA_OBJETIVO = 0.80
N_NACIONAL = 3523
N_ICA = 190

# (variable, estadistico chi2, grados de libertad) tal como figuran en la Tabla 9
FILAS = [
    ("Edad", 250.8, 4),
    ("Pobreza del hogar", 45.6, 2),
    ("Miembros del hogar", 21.2, 4),
    ("NBI2 (servicios higienicos)", 5.2, 1),
]


def potencia(w: float, n: int, gl: int) -> float:
    critico = stats.chi2.ppf(1 - ALFA, gl)
    return float(stats.ncx2.sf(critico, gl, n * w**2))


def w_minimo_detectable(n: int, gl: int) -> float:
    return float(optimize.brentq(lambda w: potencia(w, n, gl) - POTENCIA_OBJETIVO, 1e-6, 2.0))


def main() -> None:
    filas = []
    for variable, chi2, gl in FILAS:
        w = (chi2 / N_NACIONAL) ** 0.5
        filas.append({
            "variable": variable,
            "gl": gl,
            "w_cohen": round(w, 3),
            "potencia_nacional": round(potencia(w, N_NACIONAL, gl), 3),
            "potencia_ica": round(potencia(w, N_ICA, gl), 3),
        })
    minimos = {gl: round(w_minimo_detectable(N_ICA, gl), 3) for gl in (1, 2, 4)}

    for f in filas:
        print(f"{f['variable']:<30} gl={f['gl']}  w={f['w_cohen']:.3f}  "
              f"potencia nacional={f['potencia_nacional']:.3f}  potencia Ica={f['potencia_ica']:.3f}")
    print("w minimo detectable en Ica (potencia 0.80):", minimos)

    salida = {"alfa": ALFA, "n_nacional": N_NACIONAL, "n_ica": N_ICA,
              "pruebas": filas, "w_minimo_detectable_ica_potencia_080": minimos}
    (OUT_DIR / "potencia_chi2_ica.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
