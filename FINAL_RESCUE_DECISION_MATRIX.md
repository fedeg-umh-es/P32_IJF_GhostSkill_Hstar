# Final Rescue Decision Matrix — H* / Ghost Skill / Dynamic Fidelity Line

## 0. Veredicto ejecutivo

La cartera heredada es recuperable, pero no como una colección de manuscritos latentes. La arquitectura viable es una arquitectura de activos:

- `P32_IJF_GhostSkill_Hstar` queda como núcleo conceptual y repositorio de control.
- `Hstar_PM10_PM25_Madrid_Valencia` queda como núcleo empírico condicionado.
- `p34-variance-retention-api` queda como soporte metodológico común.
- Los repos de un solo caso, rank reversal, eventos horarios y prototipos H* pasan a cantera o soporte, no a núcleo editorial.

La decisión principal es no escribir ningún manuscrito nuevo hasta cerrar la trazabilidad forense del núcleo empírico Madrid-Valencia.

La línea puede aspirar a calidad Q1 solo si se reconstruye la cadena:

```text
claim -> tabla/figura -> csv/parquet -> script -> commit -> dataset
```

Actualmente esa cadena existe a nivel conceptual, pero no está cerrada a nivel de outputs versionados. El bloqueo real no es falta de idea. Es falta de trazabilidad empírica completa.

## 1. Criterio de clasificación A/B/C/D

| Clase | Significado operativo | Condición mínima |
|---|---|---|
| A — Núcleo | Activo que puede sostener la arquitectura conceptual o empírica de la nueva línea | Framework disciplinado o evidencia multi-serie trazable |
| B — Soporte | Activo reutilizable para código, contratos, métricas, tests o análisis complementario | Código o metodología reutilizable sin convertirlo en paper independiente |
| C — Cantera | Activo con piezas útiles, pero sin claims heredables como evidencia principal | Texto, figuras, scripts o decisiones recuperables con cautela |
| D — Descartar/congelar | Activo que añade más riesgo que valor | Duplicación, falta de trazabilidad, obsolescencia o riesgo editorial excesivo |

## 2. Tabla maestra final por activo

| Activo | Tipo | Estado técnico | Datos recuperables | Código ejecutable | Outputs trazables | Riesgo de leakage | Riesgo editorial | Valor estratégico | Decisión | Uso permitido |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `P32_IJF_GhostSkill_Hstar` | Repo conceptual | Parcial | N/A | No verificado como pipeline | Parcial | Bajo en diseño, no verificable en ejecución | Medio | Muy alto | A — Núcleo conceptual | Control de claims, arquitectura, matriz de evidencias |
| `Hstar_PM10_PM25_Madrid_Valencia` | Repo empírico | Condicional | Sí, documentados | Sí, scripts presentes | No cerrados en remoto | Medio hasta resolver inconsistencias | Alto por solapamiento con manuscrito enviado | Muy alto | A-minus — Núcleo empírico condicionado | Base multi-serie solo tras regenerar outputs y resolver inconsistencias |
| `p34-variance-retention-api` | Librería/pipeline | Alto | N/A | Sí, entorno y módulos presentes | Parcial | Bajo si se respeta contrato | Medio por solapamiento nominal P33/P34 | Alto | B — Soporte metodológico | Contrato común, VR/alpha/SkillVP, tests |
| `P33_variance_retention_paper` | Repo paper-first | Medio | Sí, según documentación | Sí, por diseño; no ejecutado aquí | Parcial | No verificable | Alto por solapamiento P32/P34 | Medio | B/C — Soporte técnico o cantera | Extraer implementación, no narrativa independiente |
| `P33_variance_collapse` | Paper A single-case | Medio-alto | Sí, single-location PM10 | Sí, pipeline documentado | Parcial | No verificado aquí | Muy alto por 5 rechazos/no-go | Medio | C — Cantera cerrada | Tablas, figuras, párrafos, scripts; no reenvío |
| `e2-met-validation` | Repo horario/eventos | Parcial | Sí, 3 estaciones Madrid | Parcial; Fase 1 en desarrollo | Parcial | No verificable completo | Alto con Paper A horario | Medio-alto | B/C — Soporte episódico pendiente | Eventos/P90/horario solo tras cerrar pipeline |
| `PM10-Horizons-Diagnostic` | Repo H* histórico | Medio | Sí, PM10 daily | Sí, estructura y tests declarados | Parcial | Bajo en diseño; no verificado | Alto por redundancia | Medio | C — Cantera metodológica | Documentación H*, tests, decisiones, no core |
| `madrid-pm10-rank-reversal` | Repo antecedente | Medio | Sí, Madrid Casa de Campo PM10 | Sí, pipeline declarado | Parcial | Bajo en diseño; no verificado | Alto por single-case | Bajo-medio | C — Antecedente histórico | Narrativa de rank reversal, no claim general |
| `P1_PM10_Meteorology_Hstar` | Repo meteo legado | Bajo/parcial | No confirmado | Copia de scripts base | No verificado | No verificable | Medio | Bajo-medio | C/D — Aparcar salvo necesidad meteo | Solo si aporta covariables verificables |

## 3. Auditoría del núcleo empírico Madrid-Valencia

### 3.1 Estado real

`Hstar_PM10_PM25_Madrid_Valencia` documenta un activo empírico fuerte:

- Madrid + Valencia;
- PM10 + PM2.5;
- 46 series preparadas;
- 43 series válidas;
- rolling-origin expanding-window;
- train-only preprocessing;
- simple persistence, seasonal persistence, SARIMA y XGBoost;
- horizonte `h = 1..7` días.

Sin embargo, los datos y resultados están ignorados por Git: `data_pm/`, `data_pm_daily/`, `data_processed/` y `results/` no están versionados. Por tanto, el remoto no permite verificar directamente que las tablas y figuras del manuscrito deriven de los CSV generados.

### 3.2 Predicciones fila-a-fila

El script `code/run_rolling_skill.py` sí genera predicciones por origen y horizonte en formato ancho:

```text
origin_date, horizon, actual, persist_simple, persist_seasonal, sarima, boosting
```

Esto es recuperable, pero no es todavía el contrato P32/p34.

### 3.3 Contrato requerido para P32/p34

El contrato objetivo debe ser:

```text
dataset, city, pollutant, station, model, fold, origin_date, train_end, test_start, horizon, date, y_true, y_pred
```

Mapeo necesario:

| Campo objetivo | Procedencia actual |
|---|---|
| `dataset` | reconstruible desde nombre de fichero |
| `city` | reconstruible desde nombre de fichero |
| `pollutant` | reconstruible desde nombre de fichero |
| `station` | reconstruible desde nombre de fichero |
| `model` | columnas `persist_simple`, `persist_seasonal`, `sarima`, `boosting` tras melt |
| `fold` | índice secuencial por `origin_date` dentro de serie |
| `origin_date` | presente |
| `train_end` | igual a `origin_date` por protocolo |
| `test_start` | `origin_date + 1 día` |
| `horizon` | presente |
| `date` | `origin_date + horizon días` |
| `y_true` | `actual` |
| `y_pred` | valor de la columna del modelo |

### 3.4 Inconsistencias bloqueantes antes de Q1

Hay dos inconsistencias que impiden usar el repo como evidencia Q1 sin reparación:

1. **Umbral de cobertura diaria.** La documentación afirma que se excluyen días con menos de 18 horas válidas. El script `build_daily_pm_series.py` calcula medias diarias y elimina días completamente vacíos, pero no aplica explícitamente `count >= 18`.

2. **Stride/orígenes.** La documentación puede leerse como avance diario completo, pero `run_rolling_skill.py` usa por defecto `origin_stride = 7` y `max_origins = 120`. Esto no invalida el protocolo, pero debe quedar declarado exactamente.

Hasta resolver esos dos puntos, los claims cuantitativos heredados quedan en estado condicionado.

## 4. Auditoría metodológica H*

### compliance verdict

**Conditionally compliant.**

### fatal issues

No hay una violación fatal demostrada de rolling-origin o train-only en el diseño declarado. Sí hay dos bloqueos forenses:

- la política `18 valid hourly observations/day` no está implementada en el script inspeccionado;
- la granularidad real de orígenes (`origin_stride=7`, `max_origins=120`) debe alinearse con el texto.

### softer issues

- El texto puede sobre-describir operaciones de preprocesamiento si menciona scaling cuando el script inspeccionado no lo aplica.
- Los outputs no están comprometidos en remoto.
- La reutilización futura debe controlar solapamiento con el manuscrito Madrid-Valencia ya enviado.

### minimum viable repair

1. Regenerar outputs localmente.
2. Aplicar o retirar el umbral de 18 horas válidas.
3. Documentar explícitamente stride y número máximo de orígenes.
4. Generar manifest con hashes, filas y columnas.
5. Convertir predicciones anchas al contrato largo P32/p34.

### manuscript wording guard

Hasta completar la reparación, evitar:

- “daily means with at least 18 valid hourly observations” si no se corrige el script;
- “daily-origin evaluation” si la ejecución canónica usa stride semanal;
- claims de dynamic fidelity o variance retention derivados de Madrid-Valencia si no se ha calculado alpha desde predicciones normalizadas.

## 5. Claims recuperables

| Claim heredable | Activo fuente | Evidencia actual | Evidencia faltante | Riesgo hostil | Decisión |
|---|---|---|---|---:|---|
| H* operacionaliza el forecast skill horizon frente a persistencia bajo rolling-origin | P32 + Madrid-Valencia + PM10-Horizons | Protocolo definido y usado | Trazabilidad final de outputs | Bajo-medio | Heredar |
| La evidencia multi-serie reduce fragilidad del single-case | Madrid-Valencia | 43 series documentadas | Manifest de outputs y predicciones | Bajo-medio | Heredar condicionado |
| Persistence-relative skill y dynamic fidelity pueden desacoplarse | Paper A + p34 + futuro Madrid-Valencia | Caso single-location y definición VR | VR/alpha sobre multi-serie | Medio-alto | Heredar solo como hipótesis/claim condicionado |
| Variance retention / alpha es diagnóstico auxiliar útil | p34 API + P33 retention | Código y contrato existentes | Aplicación a núcleo empírico | Bajo | Heredar |
| SkillVP es ajuste auxiliar, no métrica universal | P32 + p34 + P33 | Definición y disciplina de claims | Mantener secundario en redacción | Bajo | Heredar |
| Madrid-Valencia permite comparar ciudad, contaminante y estación | Madrid-Valencia | README/ancla/manuscrito | CSVs regenerados | Medio | Heredar condicionado |
| Valencia muestra mayor continuidad de skill que Madrid | Madrid-Valencia | Manuscrito/ancla | Reproducir tablas/figuras | Medio | Condicionado |
| PM2.5 tiene mayor continuidad de skill que PM10 | Madrid-Valencia | Manuscrito/ancla | Reproducir tablas/figuras | Medio | Condicionado |
| Rank reversal bajo rolling-origin es un patrón relevante | madrid-rank-reversal | Antecedente single-case | Replicación multi-serie | Alto | Solo antecedente |
| P75/P90 event diagnostics ayudan a evaluar utilidad operacional | Paper A + e2 | Fuerte en single-case/horario | Multi-estación o protocolo cerrado | Medio-alto | Cantera, no claim principal |
| Meteorología aporta valor marginal sobre lag-only | P1 meteo | No implementado explícitamente | Pipeline exógeno y resultados | Alto | No heredable hoy |

## 6. Matriz de solapamiento editorial

| Activos | Solapamiento | Severidad | Riesgo | Decisión de mitigación |
|---|---|---:|---|---|
| Madrid-Valencia vs manuscrito AQAH enviado | Datos, H*, ciudades, contaminantes, modelos, narrativa operacional | Muy alta | Doble publicación / salami | Cualquier uso futuro debe aportar variable nueva: dynamic fidelity/VR, no repetir H* |
| P32 vs P33 retention | Skill, alpha, SkillVP, rolling-origin | Alta | Dos papers sobre la misma idea | P32 absorbe claim; P33 queda soporte técnico |
| P32 vs Paper A collapse | Ghost Skill, VR, SkillVP | Alta | Repetición de tesis single-case | Paper A solo como caso histórico/cantera |
| Madrid-Valencia vs PM10-Horizons | H*, PM10 daily, rolling-origin | Alta | Duplicación metodológica | Madrid-Valencia como núcleo; PM10-Horizons como documentación |
| Madrid-Valencia vs rank-reversal | Madrid PM10 daily, persistence, horizon skill | Media-alta | Repetición del antecedente arXiv/single-case | Rank reversal solo antecedente narrativo |
| Paper A vs e2-met-validation | PM10 horario, episodios, P90/exceedance | Alta | Reempaquetado de paper rechazado | e2 solo para desarrollo event-based cerrado |
| p34 API vs P33 retention | Mismo contrato y VR/SkillVP | Alta | Duplicación de implementación | p34 API debe ser implementación canónica |
| P1 meteo vs Madrid-Valencia/P32 | H*, PM10, rolling-origin | Media | Abrir frente meteo sin base | Aparcar hasta que exista pipeline exógeno real |

## 7. Arquitectura mínima recuperable

| Función | Activo elegido | Razón | Condición de activación |
|---|---|---|---|
| Núcleo conceptual | `P32_IJF_GhostSkill_Hstar` | Define claims, límites, protocolo y disciplina de evidencia | Mantener como control tower, no como repo de datos |
| Núcleo empírico | `Hstar_PM10_PM25_Madrid_Valencia` | Único activo multi-ciudad/multi-pollutant suficientemente amplio | Regenerar y normalizar predicciones |
| Librería diagnóstica | `p34-variance-retention-api` | Contratos y cálculo alpha/SkillVP ya definidos | Conectar con outputs reales |
| Cantera single-case | `P33_variance_collapse` | Tablas/figuras/eventos/argumento mecanístico | No usar como paper; solo extraer piezas |
| Cantera técnica | `P33_variance_retention_paper` | Pipeline paper-first y tests | Absorber implementación si mejora p34 |
| Soporte episódico | `e2-met-validation` | P90, 3 estaciones, horario | Cerrar Fase 1 antes de claims |
| Cantera H* | `PM10-Horizons-Diagnostic` | Documentación, tests, decisiones | No usar como núcleo |
| Antecedente rank reversal | `madrid-pm10-rank-reversal` | Narrativa temprana de ranking reversal | No generalizar |

## 8. Plan de rescate por fases

### Fase 0 — Cerrada

Estado: completada.

Hechos:

- P32 definido como repositorio rector.
- `P32_SYNCHRONIZATION_PLAN.md` creado.
- `AUDIT_MASTER_TABLE.csv` creado.
- `AUDIT_HSTAR_MADRID_VALENCIA_OUTPUTS.md` creado.
- Los activos quedan clasificados A/B/C/D.

### Fase 1 — Auditoría local de outputs Madrid-Valencia

Objetivo: convertir el activo empírico de documentado a trazable.

Entregables obligatorios:

```text
outputs_manifest.csv
```

Columnas mínimas:

```text
repo, path, file_type, city, pollutant, station, artifact_type, n_rows, n_columns, columns, sha256
```

Verificaciones:

- existe un prediction CSV por cada serie válida;
- existen métricas por serie;
- existen H* summaries por serie;
- las figuras se regeneran desde los CSV;
- los resultados corresponden a 43 series válidas;
- no hay series excluidas entrando en tablas finales.

Criterio de salida:

La evidencia Madrid-Valencia pasa de A-minus a A cuando al menos una tabla principal y una figura principal se regeneran desde scripts y el manifest lo documenta.

### Fase 2 — Corrección de consistencias metodológicas

Objetivo: cerrar los dos puntos Q1-bloqueantes.

Punto 1: cobertura diaria.

Opciones válidas:

- aplicar explícitamente `count >= 18` en `build_daily_pm_series.py`; o
- retirar de README/manuscrito/ancla el claim de 18 horas; o
- demostrar que la fuente upstream ya cumple esa condición.

Punto 2: origen/stride.

Opciones válidas:

- declarar stride semanal y `max_origins=120` como diseño canónico; o
- reejecutar con stride diario si esa era la intención editorial; o
- separar configuración smoke-test de configuración paper.

Criterio de salida:

No debe quedar diferencia entre lo que dice el texto y lo que ejecuta el pipeline.

### Fase 3 — Normalización al contrato P32/p34

Objetivo: transformar predicciones anchas a predicciones largas auditables.

Archivo objetivo:

```text
p32_predictions_long.csv
```

Columnas obligatorias:

```text
dataset, city, pollutant, station, model, fold, origin_date, train_end, test_start, horizon, date, y_true, y_pred
```

Criterio de salida:

El archivo puede ser consumido por `p34-variance-retention-api` sin inferencias manuales.

### Fase 4 — Diagnóstico dynamic fidelity

Objetivo: producir evidencia interna, no manuscrito.

Archivo objetivo:

```text
p32_dynamic_fidelity_summary.csv
```

Columnas mínimas:

```text
dataset, city, pollutant, station, model, horizon, skill_rmse, alpha, skill_vp, collapse_flag, inflation_flag, near_ideal_flag
```

Criterio de salida:

Solo entonces se puede decidir qué claims de dynamic fidelity son realmente heredables.

### Fase 5 — Matriz de solapamiento con manuscritos enviados

Objetivo: evitar contaminación editorial.

Para cada claim recuperable, registrar:

- si ya aparece en el manuscrito AQAH Madrid-Valencia;
- si ya aparece en arXiv rank-reversal;
- si procede del Paper A aparcado;
- si aporta evidencia nueva o solo reempaqueta evidencia existente.

Criterio de salida:

No iniciar escritura si la diferencia con trabajos enviados es solo métrica/narrativa sin evidencia nueva trazable.

## 9. Gates finales de decisión

| Gate | Pregunta | Decisión si falla |
|---|---|---|
| G1 | ¿Hay predicciones fila-a-fila regeneradas para las 43 series válidas? | Mantener Madrid-Valencia en A-minus |
| G2 | ¿El umbral de 18 horas está implementado o retirado del texto? | No usar claims de cobertura diaria |
| G3 | ¿Stride y número de orígenes están documentados y alineados? | No usar claims de rolling-origin detallado |
| G4 | ¿Las predicciones están en contrato largo P32/p34? | No calcular VR/SkillVP multi-serie |
| G5 | ¿VR/alpha/SkillVP están calculados desde y_true/y_pred trazables? | No heredar dynamic fidelity claims |
| G6 | ¿Existe matriz de solapamiento contra manuscritos enviados? | No iniciar manuscrito |
| G7 | ¿Se puede regenerar una tabla y una figura desde scripts? | No considerar el activo Q1-ready |

## 10. No-hacer definitivo

1. No reenviar el paper Ghost Skill single-case.
2. No escribir un nuevo manuscrito desde README/ancla.
3. No crear más repos para ordenar la línea.
4. No fusionar físicamente repos antes de tener contratos de datos.
5. No promover SkillVP como métrica nueva o estándar.
6. No mezclar evidencias diarias y horarias en un mismo claim sin diseño explícito.
7. No usar Madrid-Valencia para repetir H* si ya está cubierto por manuscrito enviado.
8. No afirmar `train-only preprocessing` de operaciones que no existen en código.
9. No afirmar umbral de 18 horas si el script no lo implementa.
10. No afirmar evaluación diaria si el experimento canónico usa stride semanal.
11. No usar datos no versionados como evidencia editorial sin manifest/hash.
12. No heredar tablas manuscritas como verdad si no se regeneran desde CSV.
13. No abrir P1 meteo hasta cerrar el núcleo Madrid-Valencia + p34.
14. No usar repos C como núcleo de claims.
15. No escribir Abstract, Introduction o contribution statement antes de cerrar G1-G7.

## 11. Decisión final congelada

La cartera heredada queda así:

```text
A  P32_IJF_GhostSkill_Hstar              -> núcleo conceptual
A- Hstar_PM10_PM25_Madrid_Valencia       -> núcleo empírico condicionado
B  p34-variance-retention-api            -> soporte metodológico común
B/C P33_variance_retention_paper         -> soporte técnico o cantera
C  P33_variance_collapse                 -> cantera cerrada single-case
B/C e2-met-validation                    -> soporte episódico pendiente
C  PM10-Horizons-Diagnostic              -> cantera metodológica
C  madrid-pm10-rank-reversal             -> antecedente histórico
C/D P1_PM10_Meteorology_Hstar            -> aparcar salvo necesidad meteo
```

El único avance legítimo ahora es convertir Madrid-Valencia de A-minus a A mediante regeneración, manifest, corrección de consistencias y normalización al contrato p34.

Hasta que eso ocurra, la nueva línea Q1 existe como arquitectura recuperable, no como manuscrito.
