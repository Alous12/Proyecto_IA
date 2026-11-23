from pathlib import Path

import cv2
import numpy as np


def leer_imagen(ruta, gris=False):
    return cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_GRAYSCALE if gris else cv2.IMREAD_COLOR)


def guardar_imagen(ruta, imagen):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(ruta.suffix, imagen)[1].tofile(str(ruta))
