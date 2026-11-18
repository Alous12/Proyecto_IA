from pathlib import Path

import cv2
import numpy as np


def leer_imagen(ruta, gris=False):
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    modo = cv2.IMREAD_GRAYSCALE if gris else cv2.IMREAD_COLOR
    return cv2.imdecode(datos, modo)


def guardar_imagen(ruta, imagen):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(ruta.suffix, imagen)
    if not ok:
        raise IOError(f"No se pudo codificar la imagen: {ruta}")
    buffer.tofile(str(ruta))
