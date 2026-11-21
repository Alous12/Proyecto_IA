"""Formato de los datos procesados: manifests (CSV) y matrices (.npz)."""
import numpy as np
import pandas as pd

from src import config

COLUMNAS_INFO = ["ruta_frame", "id_video", "id_persona", "tipo_ataque", "fuente", "split"]


def leer_manifest(ruta):
    # keep_default_na=False: el split vacío "" se queda como "" y no se convierte en NaN
    return pd.read_csv(ruta, keep_default_na=False)


def fusionar_manifest(ruta, nuevo, fuentes):
    """Agrega `nuevo` al manifest de `ruta`, reemplazando solo las filas de las `fuentes` procesadas."""
    if ruta.exists():
        anterior = leer_manifest(ruta)
        nuevo = pd.concat([anterior[~anterior["fuente"].isin(fuentes)], nuevo], ignore_index=True)
    return nuevo


def nombre_matrices(fuente, tamano=config.TAMANO_REGRESION, color="gris", estadisticas_de=None):
    nombre = f"{fuente}_{tamano}_{color}"
    if estadisticas_de and estadisticas_de != fuente:
        nombre += f"_norm-{estadisticas_de}"
    return nombre


def ruta_matrices(split, fuente="lcc", tamano=config.TAMANO_REGRESION, color="gris",
                  estadisticas_de=None):
    return config.DIR_MATRICES / nombre_matrices(fuente, tamano, color, estadisticas_de) / f"{split}.npz"


def cargar_datos(split, fuente="lcc", tamano=config.TAMANO_REGRESION, color="gris",
                 estadisticas_de=None):
    ruta = ruta_matrices(split, fuente, tamano, color, estadisticas_de)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta: python -m src.data.preparar_matrices "
            f"--fuente {fuente} --tamano {tamano} --color {color}"
            + (f" --estadisticas_de {estadisticas_de}" if estadisticas_de else ""))

    datos = np.load(ruta)
    X, y = datos["X"], datos["y"]
    info = pd.DataFrame({col: datos[col] for col in COLUMNAS_INFO})
    info["etiqueta"] = y
    return X, y, info
