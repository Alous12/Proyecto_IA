import argparse
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data.imagenes import guardar_imagen

CARPETAS_LCC = {"LCC_FASD_training": ("lcc", ""), "LCC_FASD_development": ("lcc", ""),
                "LCC_FASD_evaluation": ("lcc_eval", "test")}
PATRON_LCC = re.compile(r"^(?P<video>.+_id(?P<persona>\d+)_s\d+)_\d+$")


def extraer_frames_video(ruta_video, n_frames, dir_salida):
    captura = cv2.VideoCapture(str(ruta_video))
    total = 0
    while captura.grab():
        total += 1
    captura.release()
    if total == 0:
        return []

    indices = set(np.linspace(0, total - 1, min(n_frames, total)).round().astype(int))
    rutas, indice = [], 0
    captura = cv2.VideoCapture(str(ruta_video))
    while captura.grab():
        if indice in indices:
            ok, frame = captura.retrieve()
            if ok:
                ruta = dir_salida / f"{ruta_video.stem}_f{indice:04d}.png"
                guardar_imagen(ruta, frame)
                rutas.append(ruta)
        indice += 1
    captura.release()
    return rutas


def registrar_propio(n_frames):
    filas = []
    for tipo, (etiqueta, tipo_ataque) in config.TIPOS_PROPIO.items():
        carpeta = config.DIR_PROPIO / tipo
        if not carpeta.exists():
            continue
        videos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in config.EXTENSIONES_VIDEO)
        for ruta_video in tqdm(videos, desc=f"propio/{tipo}"):
            id_persona = ruta_video.stem.rsplit("_", 1)[0]
            for ruta in extraer_frames_video(ruta_video, n_frames, config.DIR_FRAMES / "propio" / tipo):
                filas.append({"ruta_frame": config.ruta_relativa(ruta), "etiqueta": etiqueta,
                              "tipo_ataque": tipo_ataque, "id_persona": f"propio_{id_persona}",
                              "id_video": f"propio_{tipo}_{ruta_video.stem}", "fuente": "propio", "split": ""})
    return filas


def registrar_lcc(fuentes):
    filas = []
    for carpeta, (fuente, split) in CARPETAS_LCC.items():
        if fuente not in fuentes:
            continue
        for clase in ["real", "spoof"]:
            es_ataque = clase == "spoof"
            for ruta in sorted((config.DIR_LCC / carpeta / clase).glob("*.png")):
                coincidencia = PATRON_LCC.match(ruta.stem)
                filas.append({
                    "ruta_frame": config.ruta_relativa(ruta),
                    "etiqueta": config.ETIQUETA_ATAQUE if es_ataque else config.ETIQUETA_REAL,
                    "tipo_ataque": "spoof" if es_ataque else "ninguno",
                    "id_persona": f"lcc_{coincidencia['persona']}" if coincidencia else "lcc_eval_desconocido",
                    "id_video": f"lcc_{clase}_{coincidencia['video']}" if coincidencia else f"lcc_eval_{ruta.stem}",
                    "fuente": fuente,
                    "split": split,
                })
    return filas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", choices=config.FUENTES + ["todas"], default="todas")
    parser.add_argument("--n_frames", type=int, default=config.FRAMES_POR_VIDEO)
    args = parser.parse_args()

    fuentes = config.FUENTES if args.fuente == "todas" else [args.fuente]
    filas = registrar_lcc(fuentes)
    if "propio" in fuentes:
        filas += registrar_propio(args.n_frames)
    nuevo = pd.DataFrame(filas, columns=config.COLUMNAS_MANIFEST)
    if config.MANIFEST_FRAMES.exists():
        anterior = pd.read_csv(config.MANIFEST_FRAMES, keep_default_na=False)
        nuevo = pd.concat([anterior[~anterior["fuente"].isin(fuentes)], nuevo], ignore_index=True)
    config.MANIFEST_FRAMES.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(config.MANIFEST_FRAMES, index=False)
    print(nuevo.groupby(["fuente", "tipo_ataque"]).agg(frames=("ruta_frame", "size"), videos=("id_video", "nunique")))


if __name__ == "__main__":
    main()
