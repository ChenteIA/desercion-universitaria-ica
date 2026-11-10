import pandas as pd
from consolidar import (
    construir_target,
    marcar_ica,
    filtrar_edad,
    calcular_ingreso_percapita,
    extraer_perfil_jefe_hogar,
    fusionar_demografia,
    fusionar_sin_colision,
    seleccionar_columnas_finales,
    COLUMNAS_FINALES,
)


def sample_educacion():
    return pd.DataFrame({
        "CONGLOME": ["001", "001", "002", "002"],
        "VIVIENDA": ["01", "01", "01", "01"],
        "HOGAR": ["11", "11", "11", "11"],
        "CODPERSO": ["01", "02", "01", "02"],
        "P301A": ["9", "9", "10", "6"],
        "P306": ["2", "1", "2", "2"],
    })


def test_construir_target_marca_desertor():
    df = construir_target(sample_educacion())
    assert df["target"].tolist() == [1, 0]


def test_construir_target_excluye_fuera_de_poblacion():
    df = construir_target(sample_educacion())
    assert len(df) == 2  # solo P301A==9 (desertor + persistente); egresado y secundaria fuera


def test_construir_target_es_int():
    df = construir_target(sample_educacion())
    assert pd.api.types.is_integer_dtype(df["target"])


def test_construir_target_incluye_egresados_como_negativos():
    df = construir_target(sample_educacion(), incluir_egresados=True)
    assert len(df) == 3  # desertor + persistente + egresado (P301A==10); secundaria sigue fuera
    assert df["target"].tolist() == [1, 0, 0]


def test_marcar_ica_detecta_region_ica():
    df = pd.DataFrame({"UBIGEO": ["110101", "150101", "110205"]})
    resultado = marcar_ica(df)
    assert resultado["es_ica"].tolist() == [True, False, True]


def test_marcar_ica_aplica_zfill_a_ubigeo_corto():
    # UBIGEO de 5 dígitos que perdió el cero inicial: leído sin padding,
    # "11101" luce como departamento "11" (Ica) por pura coincidencia de
    # dígitos. El valor real, una vez restaurado el cero con zfill(6), es
    # "011101" -> departamento "01" (no Ica). Solo con el padding se
    # obtiene la respuesta correcta (False); sin él, este caso daría un
    # falso positivo (True).
    df = pd.DataFrame({"UBIGEO": ["11101", "150101"]})
    resultado = marcar_ica(df)
    assert resultado["es_ica"].tolist() == [False, False]


def test_filtrar_edad_incluye_bordes():
    df = pd.DataFrame({"P208A": ["16", "17", "29", "30"]})
    resultado = filtrar_edad(df)
    assert resultado["P208A"].tolist() == ["17", "29"]


def test_calcular_ingreso_percapita():
    df = pd.DataFrame({"INGHOG1D": ["3000", "1200"], "MIEPERHO": ["3", "4"]})
    resultado = calcular_ingreso_percapita(df)
    assert resultado["ingreso_percapita"].tolist() == [1000.0, 300.0]


def test_calcular_ingreso_percapita_sin_division_por_cero():
    df = pd.DataFrame({"INGHOG1D": ["1000"], "MIEPERHO": ["0"]})
    resultado = calcular_ingreso_percapita(df)
    assert pd.isna(resultado["ingreso_percapita"].iloc[0])


def sample_persona():
    return pd.DataFrame({
        "CONGLOME": ["001", "001", "002"],
        "VIVIENDA": ["01", "01", "01"],
        "HOGAR": ["11", "11", "11"],
        "CODPERSO": ["01", "02", "01"],
        "P203": ["1", "2", "1"],
    })


def sample_empleo():
    return pd.DataFrame({
        "CONGLOME": ["001", "002"],
        "VIVIENDA": ["01", "01"],
        "HOGAR": ["11", "11"],
        "CODPERSO": ["01", "01"],
        "P505": ["110", "220"],
        "P507": ["3", "2"],
    })


def test_extraer_perfil_jefe_hogar_solo_toma_al_jefe():
    df_educ = construir_target(sample_educacion())  # CODPERSO 01 y 02 en hogar 001; 01 y 02 en hogar 002
    perfil = extraer_perfil_jefe_hogar(sample_persona(), df_educ, sample_empleo())
    assert len(perfil) == 2  # un jefe por hogar (001 y 002)
    assert set(perfil.columns) == {"CONGLOME", "VIVIENDA", "HOGAR",
                                     "educacion_jefe_hogar", "ocupacion_jefe_hogar",
                                     "categoria_ocupacional_jefe"}


def test_extraer_perfil_jefe_hogar_usa_educacion_del_jefe():
    perfil = extraer_perfil_jefe_hogar(sample_persona(), sample_educacion(), sample_empleo())
    fila_hogar_001 = perfil[perfil["CONGLOME"] == "001"].iloc[0]
    assert fila_hogar_001["educacion_jefe_hogar"] == "9"


def test_merge_no_duplica_filas():
    df_educ = construir_target(sample_educacion())
    perfil_jefe = extraer_perfil_jefe_hogar(sample_persona(), sample_educacion(), sample_empleo())
    resultado = df_educ.merge(perfil_jefe, on=["CONGLOME", "VIVIENDA", "HOGAR"], how="left")
    assert len(resultado) == len(df_educ)


def test_fusionar_demografia_usa_modulo200_no_modulo300():
    # P207/P208A/P203/P209 aparecen en AMBOS df_persona (Módulo 200) y
    # df_educacion (Módulo 300, convención INEI de duplicar demográficos),
    # pero con valores deliberadamente distintos. El resultado debe tomar
    # siempre el valor de df_persona (Módulo 200), nunca el de df_educacion.
    df_persona = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"], "CODPERSO": ["01"],
        "P207": ["1"], "P208A": ["20"], "P203": ["1"], "P209": ["2"], "ESTRATO": ["3"],
    })
    df_educacion = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"], "CODPERSO": ["01"],
        "P207": ["2"], "P208A": ["99"], "P203": ["9"], "P209": ["9"],
        "P301A": ["9"], "P306": ["2"],
    })
    resultado = fusionar_demografia(df_educacion, df_persona)
    assert resultado["P207"].iloc[0] == "1"
    assert resultado["P208A"].iloc[0] == "20"
    assert resultado["P203"].iloc[0] == "1"
    assert resultado["P209"].iloc[0] == "2"


def test_fusionar_demografia_no_deja_columnas_duplicadas_x_y():
    df_persona = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"], "CODPERSO": ["01"],
        "P207": ["1"], "P208A": ["20"], "P203": ["1"], "P209": ["2"], "ESTRATO": ["3"],
    })
    df_educacion = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"], "CODPERSO": ["01"],
        "P207": ["2"], "P208A": ["99"], "P203": ["9"], "P209": ["9"],
    })
    resultado = fusionar_demografia(df_educacion, df_persona)
    assert "P207_x" not in resultado.columns
    assert "P207_y" not in resultado.columns


def test_fusionar_sin_colision_prioriza_columna_de_otro():
    # Reproduce el caso real encontrado en datos reales: el Módulo 300 (y
    # otros módulos crudos) también trae una columna llamada "FACTOR07",
    # distinta de la autoritativa que vive en Sumaria. Sin manejo explícito
    # de colisión, un merge normal produce "FACTOR07_x"/"FACTOR07_y" en vez
    # de una única "FACTOR07" limpia proveniente de la fuente correcta.
    df = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"],
        "FACTOR07": ["999"],  # valor que ya traía el módulo de origen (no autoritativo)
    })
    otro = pd.DataFrame({
        "CONGLOME": ["001"], "VIVIENDA": ["01"], "HOGAR": ["11"],
        "FACTOR07": ["123.45"],  # valor correcto de la fuente autoritativa (Sumaria)
    })
    resultado = fusionar_sin_colision(
        df, otro, ["CONGLOME", "VIVIENDA", "HOGAR", "FACTOR07"],
        on=["CONGLOME", "VIVIENDA", "HOGAR"])
    assert resultado["FACTOR07"].iloc[0] == "123.45"
    assert "FACTOR07_x" not in resultado.columns
    assert "FACTOR07_y" not in resultado.columns


def test_seleccionar_columnas_finales_excluye_columnas_no_deseadas():
    # Simula columnas crudas adicionales del Módulo 300 que no deben
    # sobrevivir a la selección final, en particular P301A y P306: son la
    # fuente de `target` y dejarlas en el CSV sería fuga de la variable
    # objetivo (label leakage) hacia las Tareas 2 y 3.
    datos = {col: [1, 2] for col in COLUMNAS_FINALES}
    datos["P300N"] = ["a", "b"]
    datos["P301A"] = ["9", "9"]
    datos["P306"] = ["1", "2"]
    datos["algo_irrelevante"] = ["x", "y"]
    df = pd.DataFrame(datos)

    resultado = seleccionar_columnas_finales(df)

    assert set(resultado.columns) == set(COLUMNAS_FINALES)
    assert "P301A" not in resultado.columns
    assert "P306" not in resultado.columns
    assert "P300N" not in resultado.columns
    assert "algo_irrelevante" not in resultado.columns


def test_marcar_area_urbana_clasifica_por_estrato():
    from consolidar import marcar_area_urbana
    df = pd.DataFrame({"ESTRATO": ["1", "5", "6", "8"]})
    resultado = marcar_area_urbana(df)
    assert resultado["area_urbana"].tolist() == [True, True, False, False]
