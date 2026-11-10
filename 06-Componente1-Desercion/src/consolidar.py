import pandas as pd
from pathlib import Path

BASE_ENAHO = (Path(__file__).parent.parent.parent / "05-Datos" / ".crudos-INEI" / "ENAHO-2024"
              / "data ENAHO" / "Condiciones de Vida y Pobreza - ENAHO" / "2024" / "Anual - (Ene-Dic)")

MOD100 = BASE_ENAHO / "966-Modulo01" / "966-Modulo01" / "Enaho01-2024-100.csv"
MOD200 = BASE_ENAHO / "966-Modulo02" / "966-Modulo02" / "Enaho01-2024-200.csv"
MOD300 = BASE_ENAHO / "966-Modulo03" / "966-Modulo03" / "Enaho01A-2024-300.csv"
MOD500 = BASE_ENAHO / "966-Modulo05" / "966-Modulo05" / "Enaho01a-2024-500.csv"
SUMARIA = BASE_ENAHO / "966-Modulo34" / "966-Modulo34" / "Sumaria-2024.csv"

OUTPUT_PATH = (Path(__file__).parent.parent.parent / "05-Datos" / "procesados"
               / "enaho_desercion_nacional.csv")
OUTPUT_PATH_SENSIBILIDAD = (Path(__file__).parent.parent.parent / "05-Datos" / "procesados"
                             / "enaho_desercion_nacional_sensibilidad.csv")

LLAVE_HOGAR = ["CONGLOME", "VIVIENDA", "HOGAR"]
LLAVE_PERSONA = LLAVE_HOGAR + ["CODPERSO"]

EDAD_MIN, EDAD_MAX = 17, 29

# Columnas demográficas que deben provenir explícitamente del Módulo 200
# (df_persona), no del Módulo 300 (df_educacion) — ver fusionar_demografia().
COLS_DEMOGRAFICOS_MOD200 = ["P207", "P208A", "P203", "P209", "ESTRATO"]

# Selección final y explícita de columnas que salen al CSV consolidado.
# Deliberadamente NO incluye P301A ni P306 (usadas solo para construir
# `target` en construir_target) ni ninguna otra columna cruda del Módulo 300
# más allá de las listadas aquí — evita fuga de la variable objetivo (label
# leakage) hacia Tareas 2 y 3.
COLUMNAS_FINALES = LLAVE_PERSONA + [
    "target", "es_ica",
    "P207", "P208A", "P203", "P209", "area_urbana", "P301D", "P300A",
    "NBI1", "NBI2", "NBI3", "NBI4", "NBI5", "P114B1",
    "educacion_jefe_hogar", "ocupacion_jefe_hogar", "categoria_ocupacional_jefe",
    "POBREZA", "MIEPERHO", "ingreso_percapita", "ESTRSOCIAL",
    "FACTOR07",
]


def construir_target(df_educacion: pd.DataFrame, incluir_egresados: bool = False) -> pd.DataFrame:
    df = df_educacion.copy()
    p301a = pd.to_numeric(df["P301A"], errors="coerce")
    p306 = pd.to_numeric(df["P306"], errors="coerce")

    es_desertor = (p301a == 9) & (p306 == 2)
    es_persistente = (p301a == 9) & (p306 == 1)
    poblacion = es_desertor | es_persistente
    if incluir_egresados:
        poblacion = poblacion | (p301a == 10)

    df = df[poblacion].copy()
    df["target"] = es_desertor[poblacion].astype(int)
    return df


def marcar_ica(df: pd.DataFrame, col_ubigeo: str = "UBIGEO") -> pd.DataFrame:
    df = df.copy()
    df["es_ica"] = df[col_ubigeo].astype(str).str.zfill(6).str[:2] == "11"
    return df


def marcar_area_urbana(df: pd.DataFrame, col_estrato: str = "ESTRATO") -> pd.DataFrame:
    df = df.copy()
    df["area_urbana"] = pd.to_numeric(df[col_estrato], errors="coerce") <= 5
    return df


def filtrar_edad(df: pd.DataFrame, col_edad: str = "P208A") -> pd.DataFrame:
    edad = pd.to_numeric(df[col_edad], errors="coerce")
    return df[(edad >= EDAD_MIN) & (edad <= EDAD_MAX)].copy()


def calcular_ingreso_percapita(df_sumaria: pd.DataFrame) -> pd.DataFrame:
    df = df_sumaria.copy()
    ingreso = pd.to_numeric(df["INGHOG1D"], errors="coerce")
    miembros = pd.to_numeric(df["MIEPERHO"], errors="coerce").replace(0, pd.NA)
    df["ingreso_percapita"] = ingreso / miembros
    return df


def extraer_perfil_jefe_hogar(df_persona: pd.DataFrame, df_educacion: pd.DataFrame,
                                df_empleo: pd.DataFrame) -> pd.DataFrame:
    jefes = df_persona[df_persona["P203"].astype(str).str.strip() == "1"][LLAVE_PERSONA].copy()

    educ_jefe = df_educacion[LLAVE_PERSONA + ["P301A"]].rename(
        columns={"P301A": "educacion_jefe_hogar"})
    empleo_jefe = df_empleo[LLAVE_PERSONA + ["P505", "P507"]].rename(
        columns={"P505": "ocupacion_jefe_hogar", "P507": "categoria_ocupacional_jefe"})

    perfil = jefes.merge(educ_jefe, on=LLAVE_PERSONA, how="left")
    perfil = perfil.merge(empleo_jefe, on=LLAVE_PERSONA, how="left")
    return perfil[LLAVE_HOGAR + ["educacion_jefe_hogar", "ocupacion_jefe_hogar",
                                   "categoria_ocupacional_jefe"]]


def fusionar_sin_colision(df: pd.DataFrame, otro: pd.DataFrame, cols: list,
                            on: list, how: str = "left") -> pd.DataFrame:
    """Hace `df.merge(otro[cols], on=on, how=how)` garantizando que las
    columnas traídas de `otro` no choquen con columnas del mismo nombre que
    `df` ya pudiera tener (p. ej. INEI duplica campos como demográficos o
    FACTOR07 entre módulos). Si `df` ya trae alguna de las columnas de
    `cols` (fuera de las llaves de merge), se descarta antes del merge, de
    modo que no se generen sufijos `_x`/`_y` y el resultado final provenga
    inequívocamente de `otro`."""
    cols_a_traer = [c for c in cols if c not in on]
    df = df.drop(columns=[c for c in cols_a_traer if c in df.columns])
    return df.merge(otro[cols], on=on, how=how)


def fusionar_demografia(df: pd.DataFrame, df_persona: pd.DataFrame) -> pd.DataFrame:
    """Agrega P207/P208A/P203/P209 desde df_persona (Módulo 200), de forma
    explícita e inequívoca. Si `df` (derivado del Módulo 300) ya trae
    columnas con esos mismos nombres (convención INEI de duplicar campos
    demográficos entre módulos), se descartan antes del merge para que no
    haya colisión `_x`/`_y` ni riesgo de que el resultado final tome el
    valor del módulo equivocado."""
    return fusionar_sin_colision(
        df, df_persona, LLAVE_PERSONA + COLS_DEMOGRAFICOS_MOD200,
        on=LLAVE_PERSONA, how="inner")


def seleccionar_columnas_finales(df: pd.DataFrame) -> pd.DataFrame:
    """Último paso antes de escribir el CSV: se queda únicamente con
    COLUMNAS_FINALES, descartando cualquier columna cruda sobrante del
    Módulo 300 (incluidas P301A y P306, la fuente de `target`)."""
    return df[COLUMNAS_FINALES].copy()


def consolidar(incluir_egresados: bool = False) -> pd.DataFrame:
    df_persona = pd.read_csv(MOD200, dtype=str, low_memory=False, encoding='latin-1')
    df_educacion = pd.read_csv(MOD300, dtype=str, low_memory=False, encoding='latin-1')
    df_vivienda = pd.read_csv(MOD100, dtype=str, low_memory=False, encoding='latin-1')
    df_empleo = pd.read_csv(MOD500, dtype=str, low_memory=False, encoding='latin-1')
    df_sumaria = pd.read_csv(SUMARIA, dtype=str, low_memory=False, encoding='latin-1')

    n_educacion_ini = len(df_educacion)

    df = construir_target(df_educacion, incluir_egresados=incluir_egresados)
    n_target = len(df)

    df = fusionar_demografia(df, df_persona)
    n_con_demografia = len(df)

    df = filtrar_edad(df)
    n_filtrado_edad = len(df)

    perfil_jefe = extraer_perfil_jefe_hogar(df_persona, df_educacion, df_empleo)
    df = fusionar_sin_colision(df, perfil_jefe, perfil_jefe.columns.tolist(), on=LLAVE_HOGAR)

    df_sumaria = calcular_ingreso_percapita(df_sumaria)
    cols_sumaria = LLAVE_HOGAR + ["POBREZA", "MIEPERHO", "ingreso_percapita", "ESTRSOCIAL",
                                    "FACTOR07"]
    df = fusionar_sin_colision(df, df_sumaria, cols_sumaria, on=LLAVE_HOGAR)

    cols_vivienda = LLAVE_HOGAR + ["NBI1", "NBI2", "NBI3", "NBI4", "NBI5", "P114B1"]
    df = fusionar_sin_colision(df, df_vivienda.drop_duplicates(subset=LLAVE_HOGAR),
                                cols_vivienda, on=LLAVE_HOGAR)

    df = marcar_ica(df)
    df = marcar_area_urbana(df)
    n_final = len(df)

    trazabilidad = {
        "personas_modulo300_inicial": n_educacion_ini,
        "con_target_construido": n_target,
        "con_demografia_modulo200": n_con_demografia,
        "filtrado_17_29": n_filtrado_edad,
        "final_consolidado": n_final,
        "desertores_nacional": int(df["target"].sum()),
        "desertores_ica": int(df.loc[df["es_ica"], "target"].sum()),
    }
    print(trazabilidad)

    df = seleccionar_columnas_finales(df)

    ruta_salida = OUTPUT_PATH_SENSIBILIDAD if incluir_egresados else OUTPUT_PATH
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)
    return df


if __name__ == "__main__":
    consolidar()
