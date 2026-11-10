# -*- coding: utf-8 -*-
"""Estimacion poblacional de la desercion universitaria (17-29) con los
factores de expansion FACTOR07 de la ENAHO 2024, conforme al compromiso de
la seccion 2.4 (estimaciones descriptivas con representatividad poblacional)."""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent
DATASET = BASE.parent / "05-Datos" / "procesados" / "enaho_desercion_nacional.csv"

df = pd.read_csv(DATASET)
df["es_ica"] = df["es_ica"].astype(bool)
df["FACTOR07"] = pd.to_numeric(df["FACTOR07"], errors="raise")

def resumen(sub, nombre):
    n = len(sub)
    n_pos = int(sub["target"].sum())
    pobl = sub["FACTOR07"].sum()
    pobl_pos = sub.loc[sub["target"] == 1, "FACTOR07"].sum()
    prev_muestral = n_pos / n
    prev_ponderada = pobl_pos / pobl
    print(f"--- {nombre} ---")
    print(f"  muestra: n={n}, desertores={n_pos}, prevalencia muestral={prev_muestral:.3f}")
    print(f"  poblacion expandida: {pobl:,.0f} personas, desertores={pobl_pos:,.0f}, "
          f"prevalencia ponderada={prev_ponderada:.3f}")

resumen(df, "Nacional (incluye Ica)")
resumen(df[df["es_ica"]], "Region Ica")
resumen(df[~df["es_ica"]], "Nacional sin Ica")
