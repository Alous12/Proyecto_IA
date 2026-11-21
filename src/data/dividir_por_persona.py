import argparse

import numpy as np
import pandas as pd

from src import config


def repartir_personas(personas, proporciones, generador):
    personas = list(personas)
    generador.shuffle(personas)
    n = len(personas)
    splits = list(proporciones)

    cantidades = {}
    for split in splits[1:]:
        cantidad = int(round(proporciones[split] * n))
        if cantidad == 0 and n >= len(splits):
            cantidad = 1
        cantidades[split] = cantidad
    cantidades[splits[0]] = n - sum(cantidades.values())

    asignacion, inicio = {}, 0
    for split in splits:
        for persona in personas[inicio:inicio + cantidades[split]]:
            asignacion[persona] = split
        inicio += cantidades[split]
    return asignacion


def dividir_estratificado(df, proporciones, semilla):
    generador = np.random.default_rng(semilla)
    clases = df.groupby("id_persona")["etiqueta"].agg(
        lambda e: "-".join(str(c) for c in sorted(set(e))))
    asignacion = {}
    for _, grupo in sorted(clases.groupby(clases)):
        asignacion.update(repartir_personas(sorted(grupo.index), proporciones, generador))
    return df["id_persona"].map(asignacion)


def verificar_sin_fuga(manifest):
    splits_por_persona = manifest.groupby(["fuente", "id_persona"])["split"].nunique()
    fugas = splits_por_persona[splits_por_persona > 1]
    if len(fugas) > 0:
        raise ValueError(f"Personas en más de un split: {list(fugas.index)}")
    print("Verificado: ninguna persona aparece en dos splits.")


def main():
    parser = argparse.ArgumentParser(description="Asigna split por persona -> manifest.csv")
    parser.add_argument("--semilla", type=int, default=config.SEMILLA)
    args = parser.parse_args()

    manifest = pd.read_csv(config.MANIFEST_ROSTROS, keep_default_na=False)

    # lcc_eval ya viene con split "test" (test externo) y no se divide
    for fuente in ["lcc", "propio"]:
        de_fuente = manifest["fuente"] == fuente
        if de_fuente.any():
            manifest.loc[de_fuente, "split"] = dividir_estratificado(
                manifest[de_fuente], config.PROPORCION_SPLITS, args.semilla)

    verificar_sin_fuga(manifest)
    manifest.to_csv(config.MANIFEST, index=False)
    print(f"Guardado {config.ruta_relativa(config.MANIFEST)}\n")

    resumen = manifest.groupby(["fuente", "split"]).agg(
        personas=("id_persona", "nunique"),
        videos=("id_video", "nunique"),
        frames=("ruta_frame", "size"),
        reales=("etiqueta", lambda e: int((e == config.ETIQUETA_REAL).sum())),
        ataques=("etiqueta", lambda e: int((e == config.ETIQUETA_ATAQUE).sum())),
    )
    print(resumen)


if __name__ == "__main__":
    main()
