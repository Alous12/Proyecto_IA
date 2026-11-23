# Detección de suplantación de identidad con fotografías en reconocimiento facial

Proyecto grupal de SIS-252 Inteligencia Artificial (UCB, Cochabamba). Clasificamos cada
video de la cámara web como **real** o **ataque** (foto impresa o foto en la pantalla de
un celular) usando varios frames por video y modelos vistos en clase, programados desde
cero con numpy: regresión logística (baseline) y red feed-forward (MLP). Además, una red
convolucional (CNN) pequeña en Keras, como en el Colab de clase.

**Convenciones:** etiqueta `0 = real`, `1 = ataque`. Todos los modelos devuelven la
probabilidad de ataque. La división train/val/test es por persona.

## 1. Instalación (Windows, VS Code)

```powershell
py -3.13 -m venv .venv            # también sirve 3.11 o 3.12
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Datos

Los datos no se suben a GitHub (ver `.gitignore`).

**LCC FASD** (Kaggle): descomprimir para que quede así:

```
data/raw/lcc_fasd/LCC_FASD/LCC_FASD_training/{real,spoof}/*.png
data/raw/lcc_fasd/LCC_FASD/LCC_FASD_development/{real,spoof}/*.png
data/raw/lcc_fasd/LCC_FASD/LCC_FASD_evaluation/{real,spoof}/*.png
```

**Dataset propio:** videos de unos 10 s por persona, con el nombre `<id_persona>_<n>.mp4`:

```
data/raw/propio/real/p01_1.mp4
data/raw/propio/impreso/p01_1.mp4
data/raw/propio/pantalla/p01_1.mp4
```

Usar el mismo `id_persona` para la misma persona en las tres carpetas (sin espacios).

**Fuentes y división:**

| Fuente | Origen | Splits |
|---|---|---|
| `lcc` | LCC FASD training + development | train/val/test por persona (70/15/15, estratificado por clase) |
| `lcc_eval` | LCC FASD evaluation | solo `test` (test externo) |
| `propio` | dataset propio | train/val/test por persona (70/15/15, estratificado por clase) |

En LCC la persona y el video salen del nombre del archivo (`AA5742_id154_s0_112.png` →
persona `154`, video `AA5742_id154_s0`). Las imágenes de evaluation no traen esos datos, así que
cada imagen cuenta como un video; por eso el test de `lcc` sale de training + development (tiene
varios frames por video y permite la métrica por video) y evaluation se usa aparte como
`lcc_eval`: imágenes que el modelo nunca vio, normalizadas con las estadísticas de train de `lcc`.
LCC no indica el tipo de ataque, así que sus ataques se marcan como `spoof`.

Datos actuales (`python -m src.data.dividir_por_persona`):

| Fuente | Split | Personas | Videos | Frames | Reales | Ataques |
|---|---|---:|---:|---:|---:|---:|
| `lcc` | train | 93 | 1,026 | 8,694 | 1,124 | 7,570 |
| `lcc` | val | 19 | 119 | 1,042 | 257 | 785 |
| `lcc` | test | 19 | 189 | 1,511 | 247 | 1,264 |
| `lcc_eval` | test | – | 7,580 | 7,580 | 314 | 7,266 |

## 3. Preprocesamiento (ejecutar desde la raíz del repo, en orden)

| Paso | Comando | Resultado |
|---|---|---|
| 1. Frames | `python -m src.data.extraer_frames --fuente todas` | `data/processed/manifest_frames.csv` (10 frames por video) |
| 2. Rostros | `python -m src.data.recortar_rostro` | `data/processed/manifest_rostros.csv`, recortes 224×224 |
| 3. División | `python -m src.data.dividir_por_persona` | `data/processed/manifest.csv` con la columna `split` |
| 4. Matrices | `python -m src.data.preparar_matrices --fuente lcc` | `data/processed/matrices/lcc_64_gris/*.npz` |
| 5. Test externo | `python -m src.data.preparar_matrices --fuente lcc_eval --estadisticas_de lcc` | `data/processed/matrices/lcc_eval_64_gris_norm-lcc/test.npz` |

`--fuente` acepta `lcc`, `lcc_eval`, `propio` o `todas`. Si solo se procesa una fuente, las filas de las otras se conservan en el manifest.
`lcc_eval` no tiene train, así que siempre necesita `--estadisticas_de lcc` (el paso 4 va antes).
Para la prueba cruzada (entrenar con LCC y evaluar con el dataset propio):

```powershell
python -m src.data.preparar_matrices --fuente propio --estadisticas_de lcc
```

Columnas de los manifests: `ruta_frame, etiqueta, tipo_ataque, id_persona, id_video, fuente, split`
(`tipo_ataque` ∈ {`ninguno`, `spoof`, `impreso`, `pantalla`}: `spoof` en LCC, `impreso`/`pantalla`
en el dataset propio; rutas relativas a la raíz, con `/`).

## 4. Usar los datos en un modelo

```python
from src.data.dataset import cargar_datos
X_train, y_train, info_train = cargar_datos("train")   # X: (N, 4096), 64x64 en gris; y: 0 real, 1 ataque
```

## 5. Evaluación

Cada modelo guarda las probabilidades de val y test:

```python
from src.evaluate import guardar_predicciones
guardar_predicciones("mlp", pd.concat([info_val, info_test]), prob_ataque)
```

y luego:

```powershell
python -m src.evaluate --modelo mlp          # results/mlp/metricas.json + gráficas
python -m src.evaluate --todos               # todos los modelos + tabla comparativa
```

El umbral se elige en validación (APCER = BPCER) y se aplica en test. Se reportan APCER,
BPCER, ACER, exactitud, AUC, APCER por tipo de ataque y matriz de confusión, por frame y
por video (promedio de la probabilidad de los frames del video).

## 6. Exploración

`notebooks/01_exploracion.ipynb`: cuántas personas, videos y frames hay por fuente y split,
frames por video, el balance de clases y ejemplos de recortes.

## 7. Modelos y entrenamiento

Modelos en `src/models/` (los tres primeros en numpy):

| Archivo | Contenido |
|---|---|
| `funciones_comunes.py` | sigmoide, ReLU y derivadas, BCE ponderada, bucle de entrenamiento con minibatches |
| `optimizadores.py` | gradient descent (lote completo), minibatch y Adam |
| `regresion_logistica.py` | baseline: `inicializar_parametros`, `calcular_logits`, `calcular_error`, `calcular_gradiente`, `actualizar_parametros`, `entrenar`, `predecir_probabilidad` |
| `mlp.py` | red feed-forward: capas ReLU con dropout, salida sigmoide, `propagacion_adelante`, `backpropagation` |
| `cnn.py` | red convolucional en Keras (`Sequential`): `construir_cnn`, `entrenar` (`model.fit` con *EarlyStopping*), `predecir_probabilidad` |

La clase real (minoritaria, ~15 % en LCC) se pondera en la BCE con `n_ataques / n_reales` de train
(`--sin_balance` lo desactiva). Siempre se guardan los parámetros de la época con menor error de validación.

```powershell
python -m src.verificar_gradiente        # backprop vs. gradiente numérico -> results/verificacion_gradiente.txt

python -m src.train --modelo regresion_logistica --optimizador adam --lr 1e-3 --lote 128
python -m src.train --modelo mlp --capas 256 64 --dropout 0.3 --optimizador adam --lr 1e-3 --paciencia 15

python -m src.experimentos               # barrido de hiperparámetros -> results/experimentos.csv
```

`src.train` guarda en `results/<nombre>/` los parámetros, `historial.csv` (error por época en train
y validación), `curva_error.png` e `hiperparametros.json`, y llama a `src.evaluate` (sección 5).
`src.experimentos` cambia un factor a la vez (optimizador, tasa de aprendizaje, tamaño de lote,
arquitectura, dropout, L2, balance), registra cada corrida en `results/experimentos.csv`/`.md`,
dibuja las curvas de validación de cada grupo en `results/experimentos/` y elige el mejor modelo
de cada tipo **por ACER en validación** (`results/mejores_modelos.json`).

Test externo (entrenar con `lcc` y evaluar con `lcc_eval`; sin `--test_todo`, porque
`lcc_eval` solo tiene test):

```powershell
python -m src.train --modelo mlp --capas 256 64 --dropout 0.3 --optimizador adam --lr 1e-3 --fuente lcc --fuente_test lcc_eval
```

Prueba cruzada (entrenar con LCC y evaluar con todo el dataset propio, normalizado con LCC):

```powershell
python -m src.data.preparar_matrices --fuente propio --estadisticas_de lcc
python -m src.train --modelo mlp --fuente_test propio --test_todo
```

### Red convolucional (CNN)

Usa los rostros a color de 64×64, así que primero hay que generar esas matrices:

```powershell
python -m src.data.preparar_matrices --fuente lcc --color rgb
python -m src.data.preparar_matrices --fuente lcc_eval --color rgb --estadisticas_de lcc
```

```powershell
python -m src.train --modelo cnn                          # -> results/cnn_64_rgb/
python -m src.train --modelo cnn --fuente_test lcc_eval   # -> results/cnn_64_rgb_test-lcc_eval/
python -m src.evaluate --todos                            # tabla comparativa con todos los modelos
```

La CNN siempre usa `color=rgb` y tamaño 64. Por defecto entrena con Adam (0.001), lotes de 128, 20 épocas
y *EarlyStopping* con paciencia 5, que recupera los pesos de la época con menor error de validación. La
clase real se pondera con `class_weight` (`n_ataques / n_reales`). En lugar de `parametros.npz` se guarda
`modelo.keras`, que se carga con `tf.keras.models.load_model`. En Windows, TensorFlow corre solo en CPU
(≈1 min). `notebooks/03_cnn.ipynb` muestra la arquitectura, las curvas, las métricas y la comparación
con los otros modelos.

`notebooks/02_entrenamiento_colab.ipynb` hace lo mismo desde Colab (subir antes
`data/processed/matrices/lcc_64_gris/` a Google Drive, y `lcc_eval_64_gris_norm-lcc/` para el
test externo).
