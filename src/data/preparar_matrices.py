import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data.imagenes import leer_imagen

COLUMNAS_INFO = ["ruta_frame", "id_video", "id_persona", "tipo_ataque", "fuente", "split"]


def nombre_matrices(fuente, tamano=64, color="gris", estadisticas_de=None):
    nombre = f"{fuente}_{tamano}_{color}"
    if estadisticas_de and estadisticas_de != fuente:
        nombre += f"_norm-{estadisticas_de}"
    return nombre


def imagen_a_vector(ruta_rel, tamano, color):
    imagen = leer_imagen(config.RAIZ / ruta_rel, gris=(color == "gris"))
    imagen = cv2.resize(imagen, (tamano, tamano), interpolation=cv2.INTER_AREA)
    if color == "rgb":
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen.astype(np.float32).ravel() / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", choices=config.FUENTES, default="lcc")
    parser.add_argument("--tamano", type=int, default=64)
    parser.add_argument("--color", choices=["gris", "rgb"], default="gris")
    parser.add_argument("--estadisticas_de", choices=config.FUENTES, default=None)
    args = parser.parse_args()
    estadisticas_de = args.estadisticas_de or args.fuente

    manifest = pd.read_csv(config.MANIFEST, keep_default_na=False)
    manifest = manifest[manifest["fuente"] == args.fuente]
    dir_salida = config.DIR_MATRICES / nombre_matrices(args.fuente, args.tamano, args.color, estadisticas_de)
    dir_salida.mkdir(parents=True, exist_ok=True)

    matrices = {}
    for split in config.SPLITS:
        filas = manifest[manifest["split"] == split]
        if len(filas) > 0:
            X = np.stack([imagen_a_vector(r, args.tamano, args.color) for r in tqdm(filas["ruta_frame"], desc=split)])
            matrices[split] = (X, filas)

    if estadisticas_de == args.fuente:
        media = matrices["train"][0].mean(axis=0)
        desviacion = matrices["train"][0].std(axis=0) + 1e-6
    else:
        normalizacion = np.load(config.DIR_MATRICES / nombre_matrices(estadisticas_de, args.tamano, args.color)
                                / "normalizacion.npz")
        media, desviacion = normalizacion["media"], normalizacion["desviacion"]
    np.savez(dir_salida / "normalizacion.npz", media=media, desviacion=desviacion, estadisticas_de=estadisticas_de)

    for split, (X, filas) in matrices.items():
        X = ((X - media) / desviacion).astype(np.float32)
        y = filas["etiqueta"].to_numpy(dtype=np.int64)
        np.savez(dir_salida / f"{split}.npz", X=X, y=y, **{c: filas[c].to_numpy(dtype=str) for c in COLUMNAS_INFO})
        print(f"{split}: X {X.shape}, ataques {y.mean():.1%}")


if __name__ == "__main__":
    main()
