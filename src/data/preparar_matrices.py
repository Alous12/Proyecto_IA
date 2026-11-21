import argparse

import cv2
import numpy as np
from tqdm import tqdm

from src import config
from src.data.dataset import COLUMNAS_INFO, leer_manifest, nombre_matrices  # también re-exporta
from src.data.imagenes import leer_imagen


def imagen_a_vector(ruta_rel, tamano, color):
    imagen = leer_imagen(config.ruta_absoluta(ruta_rel), gris=(color == "gris"))
    if imagen is None:
        raise IOError(f"No se pudo leer {ruta_rel}")
    imagen = cv2.resize(imagen, (tamano, tamano), interpolation=cv2.INTER_AREA)
    if color == "rgb":
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen.astype(np.float32).ravel() / 255.0


def construir_matriz(filas, tamano, color, descripcion):
    vectores = [imagen_a_vector(r, tamano, color)
                for r in tqdm(filas["ruta_frame"], desc=descripcion)]
    return np.stack(vectores)


def main():
    parser = argparse.ArgumentParser(description="Genera las matrices .npz normalizadas")
    parser.add_argument("--fuente", choices=config.FUENTES, default="lcc")
    parser.add_argument("--tamano", type=int, default=config.TAMANO_REGRESION)
    parser.add_argument("--color", choices=["gris", "rgb"], default="gris")
    parser.add_argument("--estadisticas_de", choices=config.FUENTES, default=None,
                        help="fuente cuyas media/desviación de train se usan (prueba cruzada)")
    args = parser.parse_args()
    estadisticas_de = args.estadisticas_de or args.fuente

    manifest = leer_manifest(config.MANIFEST)
    manifest = manifest[manifest["fuente"] == args.fuente]
    if manifest.empty:
        raise SystemExit(f"No hay filas de la fuente '{args.fuente}' en manifest.csv")
    if estadisticas_de == args.fuente and not (manifest["split"] == "train").any():
        raise SystemExit(f"La fuente '{args.fuente}' no tiene train: usa --estadisticas_de "
                         f"(por ejemplo --estadisticas_de lcc)")

    nombre = nombre_matrices(args.fuente, args.tamano, args.color, estadisticas_de)
    dir_salida = config.DIR_MATRICES / nombre
    dir_salida.mkdir(parents=True, exist_ok=True)

    matrices = {}
    for split in config.SPLITS:
        filas = manifest[manifest["split"] == split]
        if len(filas) > 0:
            matrices[split] = (construir_matriz(filas, args.tamano, args.color, split), filas)

    if estadisticas_de == args.fuente:
        X_train = matrices["train"][0]
        media = X_train.mean(axis=0)
        desviacion = X_train.std(axis=0) + 1e-6
    else:
        ruta_norm = (config.DIR_MATRICES / nombre_matrices(estadisticas_de, args.tamano, args.color)
                     / "normalizacion.npz")
        if not ruta_norm.exists():
            raise SystemExit(f"Primero genera las matrices de '{estadisticas_de}' ({ruta_norm})")
        normalizacion = np.load(ruta_norm)
        media, desviacion = normalizacion["media"], normalizacion["desviacion"]
    np.savez(dir_salida / "normalizacion.npz", media=media, desviacion=desviacion,
             estadisticas_de=estadisticas_de)

    for split, (X, filas) in matrices.items():
        X = ((X - media) / desviacion).astype(np.float32)
        y = filas["etiqueta"].to_numpy(dtype=np.int64)
        info = {col: filas[col].to_numpy(dtype=str) for col in COLUMNAS_INFO}
        np.savez(dir_salida / f"{split}.npz", X=X, y=y, **info)
        print(f"{split}: X {X.shape}, ataques {y.mean():.1%}")
    print(f"Guardado en {config.ruta_relativa(dir_salida)}")


if __name__ == "__main__":
    main()
