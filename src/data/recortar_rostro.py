import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src import config
from src.data.imagenes import guardar_imagen, leer_imagen

BASE_FUENTE = {
    "lcc": config.DIR_LCC,
    "propio": config.DIR_FRAMES / "propio",
}
ANCHO_DETECCION = 640

_detector = None


def obtener_detector():
    global _detector
    if _detector is None:
        ruta = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _detector = cv2.CascadeClassifier(ruta)
    return _detector


def detectar_rostro(imagen):
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, ANCHO_DETECCION / ancho)
    pequena = cv2.resize(imagen, None, fx=escala, fy=escala) if escala < 1 else imagen

    gris = cv2.cvtColor(pequena, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)
    lado_min = int(config.TAMANO_MIN_ROSTRO * min(gris.shape))
    rostros = obtener_detector().detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=(lado_min, lado_min))
    if len(rostros) == 0:
        return None

    x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
    return tuple(int(round(v / escala)) for v in (x, y, w, h))


def recortar_cuadrado(imagen, caja, margen):
    alto, ancho = imagen.shape[:2]
    x, y, w, h = caja
    lado = int(max(w, h) * (1 + 2 * margen))
    lado = min(lado, alto, ancho)
    cx, cy = x + w // 2, y + h // 2
    x0 = min(max(cx - lado // 2, 0), ancho - lado)
    y0 = min(max(cy - lado // 2, 0), alto - lado)
    return imagen[y0:y0 + lado, x0:x0 + lado]


def procesar_frame(ruta_frame, fuente):
    imagen = leer_imagen(config.ruta_absoluta(ruta_frame))
    if imagen is None:
        return None
    if fuente == "propio":
        caja = detectar_rostro(imagen)
        if caja is None:
            return None
        imagen = recortar_cuadrado(imagen, caja, config.MARGEN_ROSTRO)
    return cv2.resize(imagen, (config.TAMANO_ROSTRO, config.TAMANO_ROSTRO),
                      interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="Recorta rostros -> manifest_rostros.csv")
    parser.add_argument("--fuente", choices=config.FUENTES + ["todas"], default="todas")
    parser.add_argument("--sobrescribir", action="store_true",
                        help="vuelve a procesar aunque el recorte ya exista")
    args = parser.parse_args()

    frames = pd.read_csv(config.MANIFEST_FRAMES, keep_default_na=False)
    fuentes = config.FUENTES if args.fuente == "todas" else [args.fuente]
    a_procesar = frames[frames["fuente"].isin(fuentes)]

    filas, descartados = [], []
    for fila in tqdm(a_procesar.itertuples(index=False), total=len(a_procesar),
                     desc="recortando"):
        relativa = config.ruta_absoluta(fila.ruta_frame).relative_to(BASE_FUENTE[fila.fuente])
        ruta_salida = config.DIR_ROSTROS / fila.fuente / relativa.with_suffix(".png")

        if args.sobrescribir or not ruta_salida.exists():
            rostro = procesar_frame(fila.ruta_frame, fila.fuente)
            if rostro is None:
                descartados.append(fila._asdict())
                continue
            guardar_imagen(ruta_salida, rostro)

        nueva = fila._asdict()
        nueva["ruta_frame"] = config.ruta_relativa(ruta_salida)
        filas.append(nueva)

    nuevo = pd.DataFrame(filas, columns=config.COLUMNAS_MANIFEST)
    if config.MANIFEST_ROSTROS.exists():
        anterior = pd.read_csv(config.MANIFEST_ROSTROS, keep_default_na=False)
        anterior = anterior[~anterior["fuente"].isin(fuentes)]
        nuevo = pd.concat([anterior, nuevo], ignore_index=True)
    nuevo.to_csv(config.MANIFEST_ROSTROS, index=False)

    print(f"\nGuardado {config.ruta_relativa(config.MANIFEST_ROSTROS)} ({len(nuevo)} rostros)")
    if descartados:
        desc = pd.DataFrame(descartados)
        ruta_desc = config.DIR_PROCESSED / "frames_descartados.csv"
        desc.to_csv(ruta_desc, index=False)
        print(f"Frames sin rostro descartados: {len(desc)} (lista en {config.ruta_relativa(ruta_desc)})")
        print(desc.groupby(["fuente", "tipo_ataque"]).size().rename("descartados"))
    else:
        print("No se descartó ningún frame.")


if __name__ == "__main__":
    main()
