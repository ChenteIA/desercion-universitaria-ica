import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = (Path(__file__).parent.parent.parent / "05-Datos" / "procesados"
                 / "enaho_desercion_nacional.csv")
OUTPUT_PATH = (Path(__file__).parent.parent.parent / "05-Datos" / "procesados"
               / "variables_seleccionadas.json")

RANDOM_STATE = 42

CANDIDATAS = [
    "P207", "P208A", "P203", "P209",
    "P301D", "P300A",
    "NBI1", "NBI2", "NBI3", "NBI4", "NBI5", "P114B1",
    "educacion_jefe_hogar", "ocupacion_jefe_hogar", "categoria_ocupacional_jefe",
    "POBREZA", "MIEPERHO", "ingreso_percapita", "ESTRSOCIAL",
]


def filtrar_completitud(df: pd.DataFrame, candidatas: list, umbral: float = 0.4) -> list:
    return [c for c in candidatas if df[c].isna().mean() <= umbral]


def rankear_importancia(df_nacional: pd.DataFrame, candidatas: list,
                          target: str = "target") -> pd.DataFrame:
    df_train = df_nacional[~df_nacional["es_ica"]]
    X = df_train[candidatas].apply(pd.to_numeric, errors="coerce").fillna(-1)
    y = df_train[target]

    modelo = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                     class_weight="balanced", n_jobs=-1)
    modelo.fit(X, y)

    return pd.DataFrame({
        "variable": candidatas,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)


def seleccionar_variables() -> dict:
    df = pd.read_csv(DATASET_PATH)
    df["es_ica"] = df["es_ica"].astype(bool)

    completas = filtrar_completitud(df, CANDIDATAS, umbral=0.4)
    descartadas_completitud = [c for c in CANDIDATAS if c not in completas]

    ranking = rankear_importancia(df, completas)

    resultado = {
        "candidatas_iniciales": CANDIDATAS,
        "descartadas_por_completitud": descartadas_completitud,
        "seleccionadas_finales": ranking["variable"].tolist(),
        "ranking_importancia": ranking.to_dict(orient="records"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"Seleccionadas {len(ranking)}/{len(CANDIDATAS)} variables. Descartadas por "
          f"completitud: {descartadas_completitud}")
    return resultado


if __name__ == "__main__":
    seleccionar_variables()
