# Detección de suplantación de identidad con fotografías en reconocimiento facial

Proyecto grupal de SIS-252 Inteligencia Artificial (UCB). Clasifica rostros de la cámara web como
**real (0)** o **ataque (1)** (foto impresa o en pantalla) con regresión logística y red feed-forward
programadas en NumPy, y una red convolucional (CNN) en Keras. La división train/val/test es por persona.

## Instalación

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

LCC FASD (Kaggle) en `data/raw/lcc_fasd/LCC_FASD/LCC_FASD_{training,development,evaluation}/{real,spoof}/*.png`.
Dataset propio (opcional): videos `data/raw/propio/{real,impreso,pantalla}/<id_persona>_<n>.mp4`.

| Fuente | Origen | Splits |
|---|---|---|
| `lcc` | LCC training + development | train/val/test por persona (70/15/15) |
| `lcc_eval` | LCC evaluation | solo `test` (prueba externa) |
| `propio` | dataset propio | train/val/test por persona (70/15/15) |

## Preprocesamiento (en orden, desde la raíz del repo)

```powershell
python -m src.data.extraer_frames --fuente todas
python -m src.data.recortar_rostro
python -m src.data.dividir_por_persona
python -m src.data.preparar_matrices --fuente lcc
python -m src.data.preparar_matrices --fuente lcc_eval --estadisticas_de lcc
python -m src.data.preparar_matrices --fuente lcc --color rgb
python -m src.data.preparar_matrices --fuente lcc_eval --color rgb --estadisticas_de lcc
```

## Entrenamiento y evaluación

```powershell
python -m src.train --modelo regresion_logistica --optimizador minibatch --lr 0.01 --epocas 60 --paciencia 15
python -m src.train --modelo mlp --capas 256 64 --dropout 0.3 --lote 32 --epocas 60 --paciencia 15
python -m src.train --modelo cnn
python -m src.train --modelo cnn --fuente_test lcc_eval
python -m src.experimentos
python -m src.evaluate --todos
```

- `src.train` guarda en `results/<nombre>/` los parámetros (`modelo.keras` en la CNN), `historial.csv`,
  `curva_error.png`, `hiperparametros.json`, `predicciones.csv`, `metricas.json` y las gráficas.
- `--fuente_test lcc_eval` evalúa en la prueba externa; `--fuente_test propio --test_todo` en todo el dataset propio.
- `src.experimentos` hace el barrido de hiperparámetros (`results/experimentos.csv`) y elige el mejor modelo
  de cada tipo por ACER en validación (`results/mejores_modelos.json`).
- `src.evaluate --todos` reevalúa todo `results/` y arma `results/comparacion_modelos.csv`.
- El umbral se elige en validación (APCER = BPCER) y se aplica en test, por frame y por video.
