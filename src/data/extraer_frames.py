import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data.dataset import fusionar_manifest
from src.data.imagenes import guardar_imagen

# carpeta -> (fuente, split). training + development se dividen después por persona;
# evaluation es una fuente aparte (test externo).
CARPETAS_LCC = {
    "LCC_FASD_training": ("lcc", ""),
    "LCC_FASD_development": ("lcc", ""),
    "LCC_FASD_evaluation": ("lcc_eval", "test"),
}
PATRON_LCC = re.compile(r"^(?P<video>.+_id(?P<persona>\d+)_s\d+)_\d+$")


def contar_frames(ruta_video):
    captura = cv2.VideoCapture(str(ruta_video))
    total = 0
    while captura.grab():
        total += 1
    captura.release()
    return total


def extraer_frames_video(ruta_video, n_frames, dir_salida):
    total = contar_frames(ruta_video)
    if total == 0:
        print(f"  [aviso] no se pudo leer el video: {ruta_video}")
        return []

    indices = set(np.linspace(0, total - 1, min(n_frames, total)).round().astype(int))

    rutas = []
    captura = cv2.VideoCapture(str(ruta_video))
    indice = 0
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


def registrar_propio(dir_propio, n_frames):
    filas = []
    for tipo, (etiqueta, tipo_ataque) in config.TIPOS_PROPIO.items():
        carpeta = dir_propio / tipo
        if not carpeta.exists():
            print(f"  [aviso] no existe la carpeta {carpeta}")
            continue
        videos = sorted(p for p in carpeta.iterdir()
                        if p.suffix.lower() in config.EXTENSIONES_VIDEO)
        for ruta_video in tqdm(videos, desc=f"propio/{tipo}"):
            if "_" not in ruta_video.stem:
                print(f"  [aviso] nombre sin formato <id_persona>_<n>: {ruta_video.name}")
                continue
            id_persona = ruta_video.stem.rsplit("_", 1)[0]
            dir_salida = config.DIR_FRAMES / "propio" / tipo
            for ruta in extraer_frames_video(ruta_video, n_frames, dir_salida):
                filas.append({
                    "ruta_frame": config.ruta_relativa(ruta),
                    "etiqueta": etiqueta,
                    "tipo_ataque": tipo_ataque,
                    "id_persona": f"propio_{id_persona}",
                    "id_video": f"propio_{tipo}_{ruta_video.stem}",
                    "fuente": "propio",
                    "split": "",
                })
    return filas


def registrar_lcc(dir_lcc, fuentes):
    filas = []
    for carpeta, (fuente, split) in CARPETAS_LCC.items():
        if fuente not in fuentes:
            continue
        for clase in ["real", "spoof"]:
            directorio = dir_lcc / carpeta / clase
            if not directorio.exists():
                print(f"  [aviso] no existe la carpeta {directorio}")
                continue
            es_ataque = clase == "spoof"
            for ruta in sorted(directorio.glob("*.png")):
                coincidencia = PATRON_LCC.match(ruta.stem)
                if coincidencia:
                    id_persona = f"lcc_{coincidencia['persona']}"
                    id_video = f"lcc_{clase}_{coincidencia['video']}"
                else:
                    id_persona = "lcc_eval_desconocido"
                    id_video = f"lcc_eval_{ruta.stem}"
                filas.append({
                    "ruta_frame": config.ruta_relativa(ruta),
                    "etiqueta": config.ETIQUETA_ATAQUE if es_ataque else config.ETIQUETA_REAL,
                    "tipo_ataque": "spoof" if es_ataque else "ninguno",
                    "id_persona": id_persona,
                    "id_video": id_video,
                    "fuente": fuente,
                    "split": split,
                })
    return filas


def main():
    parser = argparse.ArgumentParser(description="Extrae frames y genera manifest_frames.csv")
    parser.add_argument("--fuente", choices=config.FUENTES + ["todas"], default="todas")
    parser.add_argument("--n_frames", type=int, default=config.FRAMES_POR_VIDEO,
                        help="frames por video (solo dataset propio)")
    parser.add_argument("--dir_propio", type=Path, default=config.DIR_PROPIO)
    parser.add_argument("--dir_lcc", type=Path, default=config.DIR_LCC)
    args = parser.parse_args()

    fuentes = config.FUENTES if args.fuente == "todas" else [args.fuente]
    filas = []
    if "lcc" in fuentes or "lcc_eval" in fuentes:
        filas += registrar_lcc(args.dir_lcc, fuentes)
    if "propio" in fuentes:
        filas += registrar_propio(args.dir_propio, args.n_frames)
    nuevo = fusionar_manifest(config.MANIFEST_FRAMES,
                              pd.DataFrame(filas, columns=config.COLUMNAS_MANIFEST), fuentes)

    config.MANIFEST_FRAMES.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(config.MANIFEST_FRAMES, index=False)
    print(f"\nGuardado {config.ruta_relativa(config.MANIFEST_FRAMES)} ({len(nuevo)} frames)")
    print(nuevo.groupby(["fuente", "tipo_ataque"]).agg(
        frames=("ruta_frame", "size"), videos=("id_video", "nunique")))


if __name__ == "__main__":
    main()
