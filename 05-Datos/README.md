# Datos

Los microdatos no se incluyen en este repositorio. Son públicos y gratuitos, y se descargan del portal de microdatos del Instituto Nacional de Estadística e Informática (INEI): <https://proyectos.inei.gob.pe/microdatos/>.

## Qué descargar

Encuesta Nacional de Hogares sobre Condiciones de Vida y Pobreza (ENAHO), año 2024, periodo anual (enero a diciembre), en formato CSV:

| Módulo | Contenido | Archivo que usa el código |
|---|---|---|
| 01 | Características de la vivienda y del hogar | `Enaho01-2024-100.csv` |
| 02 | Características de los miembros del hogar | `Enaho01-2024-200.csv` |
| 03 | Educación | `Enaho01A-2024-300.csv` |
| 05 | Empleo e ingresos | `Enaho01a-2024-500.csv` |
| 34 | Sumaria (indicadores calculados) | `Sumaria-2024.csv` |

## Dónde ubicarlos

`src/consolidar.py` espera esta estructura dentro de `05-Datos/`:

```
05-Datos/.crudos-INEI/ENAHO-2024/data ENAHO/Condiciones de Vida y Pobreza - ENAHO/2024/Anual - (Ene-Dic)/
  966-Modulo01/966-Modulo01/Enaho01-2024-100.csv
  966-Modulo02/966-Modulo02/Enaho01-2024-200.csv
  966-Modulo03/966-Modulo03/Enaho01A-2024-300.csv
  966-Modulo05/966-Modulo05/Enaho01a-2024-500.csv
  966-Modulo34/966-Modulo34/Sumaria-2024.csv
```

Si los archivos quedan en otra ubicación, basta ajustar la constante `BASE_ENAHO` al inicio de `src/consolidar.py`.

## Qué se genera

Al ejecutar `src/consolidar.py` se crean en `05-Datos/procesados/`:

- `enaho_desercion_nacional.csv`: base consolidada de la población de estudio.
- `enaho_desercion_nacional_sensibilidad.csv`: variante que incorpora a los egresados en la clase negativa.

Ambos archivos contienen registros individuales de la encuesta y están excluidos del control de versiones.
