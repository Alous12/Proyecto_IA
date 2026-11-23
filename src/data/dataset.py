import numpy as np
import pandas as pd

from src import config
from src.data.preparar_matrices import COLUMNAS_INFO, nombre_matrices


def cargar_datos(split, fuente="lcc", tamano=64, color="gris", estadisticas_de=None):
    datos = np.load(config.DIR_MATRICES / nombre_matrices(fuente, tamano, color, estadisticas_de) / f"{split}.npz")
    info = pd.DataFrame({col: datos[col] for col in COLUMNAS_INFO})
    info["etiqueta"] = datos["y"]
    return datos["X"], datos["y"], info
