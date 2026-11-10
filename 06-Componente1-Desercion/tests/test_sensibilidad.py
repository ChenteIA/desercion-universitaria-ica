from sensibilidad import comparar_definiciones


def test_comparar_definiciones_marca_ambos_cumplen():
    original = {"auc_roc": 0.82, "recall": 0.78}
    sensibilidad = {"auc_roc": 0.80, "recall": 0.76}
    resultado = comparar_definiciones(original, sensibilidad)
    assert resultado["original_cumple_criterios"] is True
    assert resultado["sensibilidad_cumple_criterios"] is True
    assert resultado["conclusion_estable"] is True


def test_comparar_definiciones_detecta_inestabilidad():
    original = {"auc_roc": 0.82, "recall": 0.78}
    sensibilidad = {"auc_roc": 0.65, "recall": 0.60}
    resultado = comparar_definiciones(original, sensibilidad)
    assert resultado["sensibilidad_cumple_criterios"] is False
    assert resultado["conclusion_estable"] is False
