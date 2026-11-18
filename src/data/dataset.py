import numpy as np
import pandas as pd

from src import config
from src.data.preparar_matrices import COLUMNAS_INFO, nombre_matrices


def cargar_datos(split, fuente="lcc", tamano=config.TAMANO_REGRESION, color="gris",
                 estadisticas_de=None):
    nombre = nombre_matrices(fuente, tamano, color, estadisticas_de)
    ruta = config.DIR_MATRICES / nombre / f"{split}.npz"
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
