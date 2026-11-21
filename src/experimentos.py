"""Barrido de hiperparámetros: cambia un factor a la vez respecto a una configuración base.

    python -m src.experimentos                      # todos los grupos
    python -m src.experimentos --grupo optimizador  # solo un grupo
    python -m src.experimentos --rehacer            # vuelve a correr aunque ya estén hechas

Una corrida se reutiliza solo si su hiperparametros.json tiene la misma huella de datos
(fuente, fuente_test y SHA-256 de cada .npz) que los datos actuales; si no, se vuelve a correr.
results/resumen/experimentos.csv (y .md), las figuras por grupo (results/resumen/figuras/) y
results/resumen/mejores_modelos.json se reconstruyen en cada ejecución leyendo las carpetas de
las corridas válidas (results/experimentos/<corrida>/).

El mejor modelo de cada tipo se elige por ACER en validación (no por test). Con sus parámetros
ya entrenados se predice el test externo lcc_eval (si existen sus matrices) y esas métricas se
agregan a mejores_modelos.json. Los dos modelos se copian a results/finales/<modelo>/.
"""
import argparse
import shutil

import pandas as pd

from src import config, graficas
from src.evaluate import tabla_markdown
from src.train import (completar_hiperparametros, ejecutar, evaluar_parametros, huella_datos,
                       nombre_experimento, resumen_test, sha256_archivo)

BASE_RL = {"modelo": "regresion_logistica", "optimizador": "adam", "lr": 1e-3, "lote": 128,
           "epocas": 60, "paciencia": 15}
BASE_MLP = {**BASE_RL, "modelo": "mlp", "capas": [256, 64], "dropout": 0.3}
EPOCAS_GD = 400
FUENTE_EXTERNA = "lcc_eval"


def con_epocas(h):
    if h["optimizador"] == "gd":
        return {**h, "epocas": EPOCAS_GD, "paciencia": 60}
    return h


GRUPOS = {
    "optimizador": [
        con_epocas({**base, "optimizador": opt, "lr": lr})
        for base in [BASE_RL, BASE_MLP]
        for opt in ["gd", "minibatch", "adam"]
        for lr in [1e-1, 1e-2, 1e-3, 1e-4]
    ],
    "tasa_aprendizaje": [
        {**base, "lr": lr} for base in [BASE_RL, BASE_MLP] for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    ],
    "tamano_lote": [
        {**base, "lote": lote} for base in [BASE_RL, BASE_MLP] for lote in [32, 128, 512, 2048]
    ],
    "arquitectura": [
        {**BASE_MLP, "capas": capas, "dropout": dropout}
        for capas in [[32], [128], [256, 64], [512, 128], [512, 128, 32]]
        for dropout in [0.0, 0.3]
    ],
    "dropout": [{**BASE_MLP, "dropout": d} for d in [0.0, 0.2, 0.3, 0.5]],
    "regularizacion_l2": [
        {**base, "l2": l2} for base in [BASE_RL, BASE_MLP] for l2 in [0.0, 1.0, 10.0, 100.0]
    ],
    "balance_clases": [
        {**base, "balancear": b} for base in [BASE_RL, BASE_MLP] for b in [True, False]
    ],
}


def completar(h):
    return completar_hiperparametros({"fuente_test": "lcc", **h, "verbose": False})


def corridas_del_grupo(grupo):
    """Nombres (sin repetir) y hiperparámetros de las corridas de un grupo."""
    corridas = {}
    for h in GRUPOS[grupo]:
        h = completar(h)
        corridas.setdefault(nombre_experimento(h), h)
    return corridas


def leer_corrida(nombre, huella):
    """Devuelve el hiperparametros.json de la corrida si es válida para los datos actuales.

    Devuelve None si no existe, si falta metricas.json o si su huella no coincide."""
    ruta = config.ruta_corrida(nombre, "hiperparametros.json")
    if huella is None or not ruta.exists() or not config.ruta_corrida(nombre, "metricas.json").exists():
        return None
    guardado = config.leer_json(ruta)
    return guardado if guardado.get("huella") == huella else None


def reconstruir_tabla():
    """Arma la tabla de todas las corridas válidas leyendo sus carpetas."""
    filas, nombres_validos, ignoradas, sin_correr = [], {}, set(), set()
    for grupo in GRUPOS:
        nombres_validos[grupo] = []
        for nombre, h in corridas_del_grupo(grupo).items():
            guardado = leer_corrida(nombre, huella_datos(h))
            if guardado is None:
                (ignoradas if config.ruta_corrida(nombre).exists() else sin_correr).add(nombre)
                continue
            metricas = config.leer_json(config.ruta_corrida(nombre, "metricas.json"))
            filas.append({"grupo": grupo, **guardado["resumen"], **resumen_test(metricas)})
            nombres_validos[grupo].append(nombre)
    return pd.DataFrame(filas), nombres_validos, ignoradas, sin_correr


def graficar_grupo(grupo, nombres):
    historiales = {n: pd.read_csv(config.ruta_corrida(n, "historial.csv")) for n in nombres
                   if config.ruta_corrida(n, "historial.csv").exists()}
    graficas.graficar_grupo(grupo, historiales, config.DIR_FIGURAS_GRUPOS / f"{grupo}.png")


def guardar_tabla(tabla):
    tabla = tabla.sort_values(["grupo", "modelo", "val_acer"])
    config.DIR_RESUMEN.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.RUTA_EXPERIMENTOS_CSV, index=False, float_format="%.4f")
    columnas = ["grupo", "nombre", "mejor_epoca", "error_train", "error_val", "val_acer", "val_auc",
                "test_acer_frame", "test_auc_frame", "segundos"]
    config.RUTA_EXPERIMENTOS_MD.write_text(tabla_markdown(tabla[columnas]), encoding="utf-8")


def elegir_mejores(tabla):
    mejores = {}
    for modelo, filas in tabla.drop_duplicates("nombre").groupby("modelo"):
        mejor = filas.sort_values(["val_acer", "error_val"]).iloc[0]
        mejores[modelo] = {k: (v.item() if hasattr(v, "item") else v) for k, v in mejor.items()}
    return mejores


def evaluar_test_externo(nombre):
    """Predice lcc_eval con los parámetros ya entrenados de `nombre` (no reentrena). Devuelve sus
    métricas, o None si no existen las matrices de lcc_eval."""
    guardado = config.leer_json(config.ruta_corrida(nombre, "hiperparametros.json"))
    h = {**guardado["hiperparametros"], "fuente_test": FUENTE_EXTERNA, "test_todo": False,
         "nombre": None, "verbose": False}
    huella = huella_datos(h)
    if huella is None:
        print(f"[aviso] no existen las matrices de {FUENTE_EXTERNA}; se omite el test externo de {nombre}")
        return None
    ruta_parametros = config.ruta_corrida(nombre, "parametros.npz")
    if not ruta_parametros.exists():
        # parametros.npz no se versiona en experimentos/: se regenera (misma semilla -> mismo modelo)
        ejecutar({**guardado["hiperparametros"], "verbose": False})
    huella["parametros"] = sha256_archivo(ruta_parametros)
    nombre_externo = nombre_experimento(h)
    if leer_corrida(nombre_externo, huella) is None:
        evaluar_parametros(h, ruta_parametros, nombre_externo, huella)
    metricas = config.leer_json(config.ruta_corrida(nombre_externo, "metricas.json"))
    return {"nombre": nombre_externo, **resumen_test(metricas)}


def main():
    parser = argparse.ArgumentParser(description="Barrido de hiperparámetros")
    parser.add_argument("--grupo", nargs="+", choices=list(GRUPOS), default=list(GRUPOS))
    parser.add_argument("--rehacer", action="store_true",
                        help="vuelve a correr aunque la huella de datos coincida")
    args = parser.parse_args()

    corridas_hechas = set()
    for grupo in args.grupo:
        for nombre, h in corridas_del_grupo(grupo).items():
            if nombre in corridas_hechas:
                continue
            corridas_hechas.add(nombre)
            if not args.rehacer and leer_corrida(nombre, huella_datos(h)) is not None:
                continue
            fila = ejecutar(h)
            print(f"[{grupo}] {nombre}: val ACER {fila['val_acer']:.3f}, "
                  f"test ACER {fila['test_acer_frame']:.3f} ({fila['segundos']} s)")

    tabla, nombres_validos, ignoradas, sin_correr = reconstruir_tabla()
    if ignoradas:
        print(f"\n[aviso] {len(ignoradas)} corrida(s) ignorada(s): sin huella o con datos distintos "
              f"a los actuales (vuelve a correr su grupo para actualizarlas)")
    if sin_correr:
        print(f"[aviso] {len(sin_correr)} corrida(s) de los grupos todavía sin correr")
    if tabla.empty:
        print("No hay corridas válidas para los datos actuales; no se actualizan las tablas.")
        return

    guardar_tabla(tabla)
    for grupo, nombres in nombres_validos.items():
        if nombres:
            graficar_grupo(grupo, nombres)
        else:
            (config.DIR_FIGURAS_GRUPOS / f"{grupo}.png").unlink(missing_ok=True)

    print("\nMejor configuración de cada grupo (por ACER en validación):")
    mejores_grupo = tabla.sort_values("val_acer").groupby(["grupo", "modelo"]).head(1)
    print(mejores_grupo[["grupo", "modelo", "nombre", "val_acer", "test_acer_frame"]]
          .sort_values(["grupo", "modelo"]).to_string(index=False))

    mejores = elegir_mejores(tabla)
    for fila in mejores.values():
        fila[FUENTE_EXTERNA] = evaluar_test_externo(fila["nombre"])
    config.guardar_json(config.RUTA_MEJORES_MODELOS, mejores)

    print("\nModelos finales (elegidos por validación):")
    for modelo, fila in mejores.items():
        # Copia la corrida elegida (y su test externo, si existe) a results/finales/<modelo>/
        destino = config.ruta_final(modelo)
        shutil.rmtree(destino, ignore_errors=True)
        shutil.copytree(config.ruta_corrida(fila["nombre"]), destino)
        externo = fila[FUENTE_EXTERNA]
        if externo:
            shutil.copytree(config.ruta_corrida(externo["nombre"]), destino / FUENTE_EXTERNA)
        texto_externo = (f"  {FUENTE_EXTERNA} ACER {externo['test_acer_frame']:.3f} "
                         f"AUC {externo['test_auc_frame']:.3f}" if externo else "")
        print(f"  {modelo}: {fila['nombre']}  val ACER {fila['val_acer']:.3f}  "
              f"test ACER {fila['test_acer_frame']:.3f}  test AUC {fila['test_auc_frame']:.3f}{texto_externo}")
    print(f"\nGuardado en {config.ruta_relativa(config.DIR_RESUMEN)}/ (tablas, figuras y "
          f"mejores_modelos.json, {len(tabla)} filas) y {config.ruta_relativa(config.DIR_FINALES)}/")


if __name__ == "__main__":
    main()
