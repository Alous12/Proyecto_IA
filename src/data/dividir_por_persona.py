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
        cantidades[split] = 1 if cantidad == 0 and n >= len(splits) else cantidad
    cantidades[splits[0]] = n - sum(cantidades.values())

    asignacion, inicio = {}, 0
    for split in splits:
        for persona in personas[inicio:inicio + cantidades[split]]:
            asignacion[persona] = split
        inicio += cantidades[split]
    return asignacion


def dividir_estratificado(df, proporciones, semilla):
    generador = np.random.default_rng(semilla)
    clases = df.groupby("id_persona")["etiqueta"].agg(lambda e: "-".join(str(c) for c in sorted(set(e))))
    asignacion = {}
    for _, grupo in sorted(clases.groupby(clases)):
        asignacion.update(repartir_personas(sorted(grupo.index), proporciones, generador))
    return df["id_persona"].map(asignacion)


def main():
    manifest = pd.read_csv(config.MANIFEST_ROSTROS, keep_default_na=False)
    for fuente in ["lcc", "propio"]:
        de_fuente = manifest["fuente"] == fuente
        if de_fuente.any():
            manifest.loc[de_fuente, "split"] = dividir_estratificado(
                manifest[de_fuente], config.PROPORCION_SPLITS, config.SEMILLA)
    manifest.to_csv(config.MANIFEST, index=False)
    print(manifest.groupby(["fuente", "split"]).agg(
        personas=("id_persona", "nunique"), videos=("id_video", "nunique"), frames=("ruta_frame", "size"),
        ataques=("etiqueta", "sum")))


if __name__ == "__main__":
    main()
