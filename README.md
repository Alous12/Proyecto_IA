# Detección de suplantación de identidad con fotografías en reconocimiento facial

Proyecto grupal de SIS-252 Inteligencia Artificial (UCB, Cochabamba). Clasificamos cada
video de la cámara web como **real** o **ataque** (foto impresa o foto en la pantalla de
un celular) usando varios frames por video y modelos vistos en clase, programados desde
cero con numpy: regresión logística (baseline) y red feed-forward (MLP).

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

## 3. Preprocesamiento (ejecutar desde la raíz del repo, en orden)

| Paso | Comando | Resultado |
|---|---|---|
| 1. Frames | `python -m src.data.extraer_frames --fuente todas` | `data/processed/manifest_frames.csv` (10 frames por video) |
| 2. Rostros | `python -m src.data.recortar_rostro` | `data/processed/manifest_rostros.csv`, recortes 224×224 |
| 3. División | `python -m src.data.dividir_por_persona` | `data/processed/manifest.csv` con la columna `split` |
| 4. Matrices | `python -m src.data.preparar_matrices --fuente lcc` | `data/processed/matrices/lcc_64_gris/*.npz` |

`--fuente` acepta `lcc`, `propio` o `todas`. Si solo se procesa una fuente, las filas de la otra se conservan en el manifest.
Para la prueba cruzada (entrenar con LCC y evaluar con el dataset propio):

```powershell
python -m src.data.preparar_matrices --fuente propio --estadisticas_de lcc
```

Columnas de los manifests: `ruta_frame, etiqueta, tipo_ataque, id_persona, id_video, fuente, split`
(`tipo_ataque` ∈ {`ninguno`, `impreso`, `pantalla`}; rutas relativas a la raíz, con `/`).

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

`notebooks/01_exploracion.ipynb`: cuántas personas, videos y frames hay por split y por
tipo, el balance de clases y ejemplos de recortes.

## 7. Modelos y entrenamiento

Modelos en numpy, en `src/models/`:

| Archivo | Contenido |
|---|---|
| `funciones_comunes.py` | sigmoide, ReLU y derivadas, BCE ponderada, bucle de entrenamiento con minibatches |
| `optimizadores.py` | gradient descent (lote completo), minibatch y Adam |
| `regresion_logistica.py` | baseline: `inicializar_parametros`, `calcular_logits`, `calcular_error`, `calcular_gradiente`, `actualizar_parametros`, `entrenar`, `predecir_probabilidad` |
| `mlp.py` | red feed-forward: capas ReLU con dropout, salida sigmoide, `propagacion_adelante`, `backpropagation` |

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

Prueba cruzada (entrenar con LCC y evaluar con todo el dataset propio, normalizado con LCC):

```powershell
python -m src.data.preparar_matrices --fuente propio --estadisticas_de lcc
python -m src.train --modelo mlp --fuente_test propio --test_todo
```

`notebooks/02_entrenamiento_colab.ipynb` hace lo mismo desde Colab (subir antes
`data/processed/matrices/lcc_64_gris/` a Google Drive).
