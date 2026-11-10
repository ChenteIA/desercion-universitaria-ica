"""EDA descriptivo ampliado del Cap. III: departamento, edad/sexo y 19 variables.

Anclas de validacion (cifras YA publicadas en el Informe, deben reproducirse):
  - Tabla 1: 3,523 personas, 888 desertores (25.2%); Ica 190/42 (22.1%)
  - Tabla 3: expandido nacional 1,210,245 / 311,593 (25.7%); Ica 31,730 / 8,021
  - Tabla 4 actual: solteria 95.5/78.3, conviviente 3.0/15.1, separado 1.1/5.2,
    internet 51.9/43.5, pobreza 6.6/13.9 (persistentes/desertoras)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TESIS = Path(__file__).parent.parent.parent
CSV = TESIS / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"
MOD300 = (TESIS / "05-Datos" / ".crudos-INEI" / "ENAHO-2024" / "data ENAHO"
          / "Condiciones de Vida y Pobreza - ENAHO" / "2024" / "Anual - (Ene-Dic)"
          / "966-Modulo03" / "966-Modulo03" / "Enaho01A-2024-300.csv")
TABLAS = TESIS / "08-Resultados" / "Tablas" / "productivo"
FIGURAS = TESIS / "08-Resultados" / "Figuras" / "productivo"

LLAVE = ["CONGLOME", "VIVIENDA", "HOGAR", "CODPERSO"]

DEPARTAMENTOS = {
    "01": "Amazonas", "02": "Áncash", "03": "Apurímac", "04": "Arequipa",
    "05": "Ayacucho", "06": "Cajamarca", "07": "Callao", "08": "Cusco",
    "09": "Huancavelica", "10": "Huánuco", "11": "Ica", "12": "Junín",
    "13": "La Libertad", "14": "Lambayeque", "15": "Lima", "16": "Loreto",
    "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes",
    "25": "Ucayali",
}

TRAMOS = [(17, 19), (20, 21), (22, 24), (25, 29)]


def cargar_con_departamento() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={k: str for k in LLAVE})
    ubigeo = pd.read_csv(MOD300, dtype=str, low_memory=False, encoding="latin-1",
                         usecols=LLAVE + ["UBIGEO"]).drop_duplicates(subset=LLAVE)
    df = df.merge(ubigeo, on=LLAVE, how="left", validate="one_to_one")
    assert len(df) == 3523, f"merge altero el numero de filas: {len(df)}"
    assert df["UBIGEO"].notna().all(), "UBIGEO nulo tras el merge"
    df["dep"] = df["UBIGEO"].astype(str).str.zfill(6).str[:2]
    df["departamento"] = df["dep"].map(DEPARTAMENTOS)
    assert df["departamento"].notna().all(), "codigo de departamento fuera del mapa"
    es_ica_csv = df["es_ica"].astype(str).isin(["True", "1", "1.0"])
    assert (es_ica_csv == (df["dep"] == "11")).all(), "es_ica no cuadra con UBIGEO"
    df["FACTOR07"] = pd.to_numeric(df["FACTOR07"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype(int)
    return df


def tabla_departamentos(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("departamento").apply(
        lambda x: pd.Series({
            "n": len(x),
            "desertores": int(x["target"].sum()),
            "prev_muestral": 100 * x["target"].mean(),
            "poblacion_exp": x["FACTOR07"].sum(),
            "desertores_exp": (x["FACTOR07"] * x["target"]).sum(),
        }), include_groups=False)
    g["prev_ponderada"] = 100 * g["desertores_exp"] / g["poblacion_exp"]
    g = g.sort_values("prev_ponderada", ascending=False).reset_index()
    total = pd.DataFrame([{
        "departamento": "Total nacional", "n": len(df),
        "desertores": int(df["target"].sum()),
        "prev_muestral": 100 * df["target"].mean(),
        "poblacion_exp": df["FACTOR07"].sum(),
        "desertores_exp": (df["FACTOR07"] * df["target"]).sum(),
        "prev_ponderada": 100 * (df["FACTOR07"] * df["target"]).sum() / df["FACTOR07"].sum(),
    }])
    out = pd.concat([g, total], ignore_index=True)
    # anclas Tabla 3
    assert abs(total.iloc[0]["poblacion_exp"] - 1_210_245) < 1000, total.iloc[0]["poblacion_exp"]
    assert abs(total.iloc[0]["desertores_exp"] - 311_593) < 1000, total.iloc[0]["desertores_exp"]
    ica = g[g["departamento"] == "Ica"].iloc[0]
    assert int(ica["n"]) == 190 and int(ica["desertores"]) == 42, "Ica no cuadra con Tabla 1"
    out.to_csv(TABLAS / "desercion_por_departamento.csv", index=False, encoding="utf-8")
    return out


def tabla_edad_sexo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["edad"] = pd.to_numeric(df["P208A"], errors="coerce")
    df["sexo"] = pd.to_numeric(df["P207"], errors="coerce").map({1: "Hombre", 2: "Mujer"})
    filas = []
    for lo, hi in TRAMOS:
        sub = df[(df["edad"] >= lo) & (df["edad"] <= hi)]
        for etiqueta, s in [("Total", sub), ("Hombre", sub[sub["sexo"] == "Hombre"]),
                            ("Mujer", sub[sub["sexo"] == "Mujer"])]:
            filas.append({
                "tramo": f"{lo}-{hi}", "sexo": etiqueta, "n": len(s),
                "desertores": int(s["target"].sum()),
                "prev_muestral": 100 * s["target"].mean(),
                "prev_ponderada": 100 * (s["FACTOR07"] * s["target"]).sum() / s["FACTOR07"].sum(),
            })
    out = pd.DataFrame(filas)
    assert out[out["sexo"] == "Total"]["n"].sum() == 3523
    out.to_csv(TABLAS / "prevalencia_edad_sexo.csv", index=False, encoding="utf-8")
    return out


def tabla_19_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numericas = {"P208A": "Edad (años)", "MIEPERHO": "Miembros del hogar",
                 "ingreso_percapita": "Ingreso per cápita anual del hogar (S/)"}
    filas = []
    grupos = {"Persistentes": df[df["target"] == 0], "Desertoras": df[df["target"] == 1]}
    for var, nombre in numericas.items():
        fila = {"variable": var, "descripcion": nombre, "tipo": "numerica", "categoria": "media (DE)"}
        for g, sub in grupos.items():
            v = pd.to_numeric(sub[var], errors="coerce")
            fila[g] = f"{v.mean():.1f} ({v.std():.1f})"
        filas.append(fila)
    categoricas = ["P207", "P203", "P209", "P301D", "P300A", "P114B1",
                   "NBI1", "NBI2", "NBI3", "NBI4", "NBI5", "POBREZA",
                   "ESTRSOCIAL", "educacion_jefe_hogar", "ocupacion_jefe_hogar",
                   "categoria_ocupacional_jefe"]
    for var in categoricas:
        serie = df[var].astype(str).str.strip().replace("", np.nan)
        for cat in sorted(serie.dropna().unique()):
            fila = {"variable": var, "descripcion": var, "tipo": "categorica", "categoria": cat}
            for g, sub in grupos.items():
                s = sub[var].astype(str).str.strip().replace("", np.nan)
                fila[g] = f"{100 * (s == cat).sum() / s.notna().sum():.1f}"
            filas.append(fila)
    out = pd.DataFrame(filas)
    # anclas Tabla 4 actual (P209: 6=soltero; POBREZA 1|2 = pobre)
    def pct(g, var, cats):
        s = grupos[g][var].astype(str).str.strip().replace("", np.nan)
        return 100 * s.isin(cats).sum() / s.notna().sum()
    for g, esperado in [("Persistentes", 95.5), ("Desertoras", 78.3)]:
        assert abs(pct(g, "P209", ["6", "6.0"]) - esperado) < 0.15, f"solteria {g}: {pct(g, 'P209', ['6', '6.0']):.1f}"
    for g, esperado in [("Persistentes", 6.6), ("Desertoras", 13.9)]:
        assert abs(pct(g, "POBREZA", ["1", "2", "1.0", "2.0"]) - esperado) < 0.15, f"pobreza {g}: {pct(g, 'POBREZA', ['1', '2', '1.0', '2.0']):.1f}"
    out.to_csv(TABLAS / "eda_19_variables.csv", index=False, encoding="utf-8")
    return out


def figura_departamentos(t: pd.DataFrame) -> None:
    d = t[t["departamento"] != "Total nacional"].sort_values("prev_ponderada")
    prev_nac = t[t["departamento"] == "Total nacional"]["prev_ponderada"].iloc[0]
    colores = ["#2a6f97" if x == "Ica" else "#61a5c2" for x in d["departamento"]]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.barh(d["departamento"], d["prev_ponderada"], color=colores)
    ax.axvline(prev_nac, color="gray", ls="--", lw=1,
               label=f"Nacional ({prev_nac:.1f}%)")
    ax.set_xlabel("Prevalencia ponderada de deserción (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS / "prevalencia_departamentos.png", dpi=150)
    plt.close(fig)


def figura_edad_sexo(t: pd.DataFrame) -> None:
    tramos = [f"{lo}-{hi}" for lo, hi in TRAMOS]
    x = np.arange(len(tramos)); ancho = 0.27
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (grupo, color) in enumerate([("Total", "#2a6f97"), ("Hombre", "#61a5c2"),
                                        ("Mujer", "#a9cce3")]):
        vals = [t[(t["tramo"] == tr) & (t["sexo"] == grupo)]["prev_ponderada"].iloc[0]
                for tr in tramos]
        ax.bar(x + (i - 1) * ancho, vals, ancho, label=grupo, color=color)
    ax.set_xticks(x); ax.set_xticklabels(tramos)
    ax.set_xlabel("Tramo etario (años)", fontsize=9)
    ax.set_ylabel("Prevalencia ponderada de deserción (%)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "prevalencia_edad_sexo.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = cargar_con_departamento()
    td = tabla_departamentos(df)
    te = tabla_edad_sexo(df)
    tabla_19_variables(df)
    figura_departamentos(td)
    figura_edad_sexo(te)
    print("OK — CSVs y PNGs generados; todas las anclas cuadraron")
