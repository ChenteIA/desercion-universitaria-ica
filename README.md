# Predicción del riesgo de deserción universitaria en la región Ica

Código fuente de la tesis **"Predicción del riesgo de deserción universitaria mediante *Random Forest* y datos socioeconómicos de la Encuesta Nacional de Hogares en la región Ica, 2024"**, presentada para optar el título profesional de Licenciado en Matemática e Informática en la Universidad Nacional San Luis Gonzaga (Ica, Perú).

Autor: Vicente Jesús Enrique Celis Arias.

## Qué hace

Entrena un modelo *Random Forest* con datos de alcance nacional de la Encuesta Nacional de Hogares (ENAHO) 2024 y lo valida en la región Ica, que queda excluida de todo el ajuste. Lo compara con una regresión logística y con XGBoost, calibra el umbral de decisión, examina la equidad entre subgrupos y explica las predicciones con valores SHAP.

- Población: personas de 17 a 29 años con estudios universitarios sin concluir (3,523 registros; 888 casos de deserción, 42 de ellos en Ica).
- Resultado principal: área bajo la curva ROC de 0.716 en Ica (IC 95%: 0.624–0.802), por debajo del criterio de 0.80 fijado en la hipótesis general.

## Estructura

```
05-Datos/
  README.md                      cómo obtener los microdatos (no se redistribuyen)
  procesados/
    variables_seleccionadas.json variables candidatas y seleccionadas
06-Componente1-Desercion/
  src/                           tubería principal, un módulo por fase
    consolidar.py                integra los módulos de la ENAHO y construye la variable objetivo
    seleccionar.py               selección de variables
    entrenar.py                  entrenamiento, búsqueda de hiperparámetros y calibración del umbral
    evaluar.py                   métricas, intervalos de confianza y equidad
    interpretar.py               valores SHAP globales e individuales
    sensibilidad.py              réplica con una definición alternativa de la clase negativa
  tests/                         pruebas automatizadas de cada módulo
  analisis/                      análisis complementarios del informe
    salidas/                     resultados agregados en JSON
  requirements.txt               versiones exactas de las bibliotecas
```

## Cómo reproducir

1. Instalar las dependencias (Python 3.12 o superior):

   ```
   pip install -r 06-Componente1-Desercion/requirements.txt
   ```

2. Descargar los microdatos de la ENAHO 2024 y ubicarlos como indica [`05-Datos/README.md`](05-Datos/README.md).

3. Ejecutar la tubería, en este orden, desde `06-Componente1-Desercion/`:

   ```
   python src/consolidar.py
   python src/seleccionar.py
   python src/entrenar.py
   python src/evaluar.py
   python src/interpretar.py
   python src/sensibilidad.py
   ```

   Los modelos se guardan en `06-Componente1-Desercion/modelos/` y las tablas y figuras en `08-Resultados/`. Ambas carpetas se generan localmente y no se versionan.

4. Los análisis complementarios se ejecutan de forma independiente, una vez entrenados los modelos. Por ejemplo:

   ```
   python analisis/ic_auc_subgrupos.py
   python analisis/recalibracion_rf.py
   ```

Las pruebas no requieren los microdatos:

```
cd 06-Componente1-Desercion
pytest
```

Todos los procedimientos aleatorios usan la semilla 42.

## Datos

Los microdatos pertenecen al Instituto Nacional de Estadística e Informática (INEI) y son de acceso público y gratuito. Este repositorio no los incluye, ni incluye los modelos entrenados ni registros individuales de personas encuestadas; solo contiene código y resultados agregados.

## Advertencia de uso

El puntaje que produce el modelo es un instrumento de priorización del apoyo institucional. No debe emplearse para sancionar, estigmatizar ni restringir servicios a ninguna persona. El modelo identifica perfiles asociados a una deserción ya ocurrida, con datos transversales; su uso como sistema de alerta temprana requeriría una validación previa con datos longitudinales.
