"""Entrena un modelo, guarda parámetros, historial y curva, y lo evalúa con src.evaluate.

    python -m src.train --modelo regresion_logistica --optimizador adam --lr 1e-3 --lote 128
    python -m src.train --modelo mlp --capas 256 64 --dropout 0.3 --optimizador adam --lr 1e-3

Prueba cruzada (entrenar con LCC, evaluar con todo el dataset propio):
    python -m src.train --modelo mlp --fuente_test propio --test_todo

Salida en results/experimentos/<nombre>/: parametros.npz, historial.csv, curva_error.png,
hiperparametros.json, predicciones.csv, metricas.json y las gráficas de evaluate.
"""
import argparse
import hashlib
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config, graficas
from src.data.dataset import cargar_datos, ruta_matrices
from src.evaluate import evaluar_modelo, guardar_predicciones
from src.metrics import calcular_metricas, elegir_umbral
from src.models import mlp, regresion_logistica
from src.models.funciones_comunes import calcular_peso_real
from src.models.optimizadores import OPTIMIZADORES, crear_optimizador

MODELOS = {"regresion_logistica": regresion_logistica, "mlp": mlp}
ABREVIATURA = {"regresion_logistica": "rl", "mlp": "mlp"}

# Único lugar con los valores por defecto. clave: (valor por defecto, opciones de argparse).
# "bandera" cambia el nombre de la opción (p. ej. --sin_balance guarda False en "balancear").
HIPERPARAMETROS = {
    "modelo": ("regresion_logistica", {"choices": list(MODELOS)}),
    "capas": ([256, 64], {"type": int, "nargs": "+", "help": "neuronas por capa oculta (solo mlp)"}),
    "dropout": (0.0, {"type": float, "help": "probabilidad de apagar una neurona oculta (solo mlp)"}),
    "optimizador": ("adam", {"choices": list(OPTIMIZADORES)}),
    "lr": (1e-3, {"type": float, "help": "tasa de aprendizaje"}),
    "lote": (128, {"type": int, "help": "tamaño de minibatch (no aplica a gd)"}),
    "epocas": (30, {"type": int}),
    "paciencia": (None, {"type": int, "help": "épocas sin mejorar en validación antes de parar"}),
    "l2": (0.0, {"type": float, "help": "regularización L2"}),
    "balancear": (True, {"bandera": "--sin_balance", "action": "store_false",
                         "help": "no ponderar la clase real en la BCE"}),
    "fuente": ("lcc", {"choices": config.FUENTES}),
    "fuente_test": (None, {"choices": config.FUENTES,
                           "help": "fuente del test (prueba cruzada); por defecto la misma"}),
    "test_todo": (False, {"action": "store_true",
                          "help": "en prueba cruzada usa train+val+test de fuente_test como test"}),
    "tamano": (config.TAMANO_REGRESION, {"type": int}),
    "color": ("gris", {"choices": ["gris", "rgb"]}),
    "semilla": (config.SEMILLA, {"type": int}),
    "nombre": (None, {"help": "carpeta en results/experimentos/ (por defecto se genera)"}),
    "verbose": (True, {"bandera": "--silencioso", "action": "store_false"}),
}
HIPERPARAMETROS_POR_DEFECTO = {clave: defecto for clave, (defecto, _) in HIPERPARAMETROS.items()}


def nombre_experimento(h):
    partes = [ABREVIATURA[h["modelo"]]]
    if h["modelo"] == "mlp":
        partes.append("-".join(str(c) for c in h["capas"]))
    partes += [h["optimizador"], f"lr{h['lr']:g}"]
    if h["optimizador"] != "gd":
        partes.append(f"b{h['lote']}")
    if h["modelo"] == "mlp":
        partes.append(f"do{h['dropout']:g}")
    if h["l2"]:
        partes.append(f"l2{h['l2']:g}")
    if not h["balancear"]:
        partes.append("sinbal")
    if h["fuente_test"] != h["fuente"]:
        partes.append(f"test-{h['fuente_test']}")
    return "_".join(partes)


def matrices_usadas(h):
    """(split, fuente, estadisticas_de) de cada .npz que se carga para train, val y test."""
    # Prueba cruzada: el test sale de otra fuente, normalizada con las estadísticas de train
    cruzada = h["fuente_test"] != h["fuente"]
    splits_test = config.SPLITS if cruzada and h["test_todo"] else ["test"]
    return {"train": [("train", h["fuente"], None)], "val": [("val", h["fuente"], None)],
            "test": [(s, h["fuente_test"], h["fuente"] if cruzada else None) for s in splits_test]}


@lru_cache(maxsize=None)
def _sha256(ruta, _tamano, _modificado):
    with open(ruta, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sha256_archivo(ruta):
    """SHA-256 de un archivo (se recalcula solo si cambia su tamaño o su fecha de modificación)."""
    info = ruta.stat()
    return _sha256(str(ruta), info.st_size, info.st_mtime_ns)


def huella_datos(h):
    """Identifica los datos exactos de una corrida: fuente, fuente_test y SHA-256 de cada .npz."""
    archivos = {}
    for conjunto, lista in matrices_usadas(h).items():
        for split, fuente, estadisticas_de in lista:
            ruta = ruta_matrices(split, fuente, h["tamano"], h["color"], estadisticas_de)
            if not ruta.exists():
                return None
            archivos[config.ruta_relativa(ruta)] = sha256_archivo(ruta)
    return {"fuente": h["fuente"], "fuente_test": h["fuente_test"], "test_todo": h["test_todo"],
            "sha256": archivos}


def cargar_conjunto(h, conjunto):
    """Carga train, val o test. En prueba cruzada el test puede juntar varios .npz de otra fuente."""
    partes = [cargar_datos(s, fuente, h["tamano"], h["color"], estadisticas_de=est)
              for s, fuente, est in matrices_usadas(h)[conjunto]]
    info = pd.concat([p[2] for p in partes], ignore_index=True)
    info["split"] = conjunto
    return np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes]), info


def entrenar_modelo(h, X, y, X_val, y_val, peso_real):
    optimizador = crear_optimizador(h["optimizador"], h["lr"])
    tam_lote = len(y) if optimizador.usa_lote_completo else h["lote"]
    comunes = dict(epocas=h["epocas"], tam_lote=tam_lote, peso_real=peso_real, l2=h["l2"],
                   paciencia=h["paciencia"], semilla=h["semilla"], verbose=h["verbose"])
    if h["modelo"] == "mlp":
        return mlp.entrenar(X, y, X_val, y_val, optimizador, capas_ocultas=h["capas"],
                            prob_dropout=h["dropout"], **comunes)
    return regresion_logistica.entrenar(X, y, X_val, y_val, optimizador, **comunes)


def completar_hiperparametros(h):
    h = {**HIPERPARAMETROS_POR_DEFECTO, **h}
    h["capas"] = [int(c) for c in h["capas"]]
    h["fuente_test"] = h["fuente_test"] or h["fuente"]
    return h


def resumen_test(metricas):
    return {f"test_{clave}_{nivel.replace('por_', '')}": metricas[nivel][clave]
            for nivel in ["por_frame", "por_video"] for clave in ["apcer", "bpcer", "acer", "auc"]}


def ejecutar(h):
    """Entrena y evalúa un experimento descrito por el diccionario h. Devuelve un resumen."""
    h = completar_hiperparametros(h)
    nombre = h.get("nombre") or nombre_experimento(h)
    directorio = config.ruta_corrida(nombre)
    directorio.mkdir(parents=True, exist_ok=True)
    print(f"\n### {nombre}")

    (X, y, _), (X_val, y_val, info_val), (X_test, _, info_test) = (
        cargar_conjunto(h, conjunto) for conjunto in ["train", "val", "test"])
    peso_real = calcular_peso_real(y) if h["balancear"] else 1.0

    inicio = time.time()
    params, historial, mejor_epoca = entrenar_modelo(h, X, y, X_val, y_val, peso_real)
    segundos = time.time() - inicio

    modelo = MODELOS[h["modelo"]]
    prob_val = modelo.predecir_probabilidad(X_val, params)
    prob_test = modelo.predecir_probabilidad(X_test, params)

    np.savez(directorio / "parametros.npz", **params)
    pd.DataFrame(historial).to_csv(directorio / "historial.csv", index=False, float_format="%.6f")
    graficas.graficar_curva(historial, mejor_epoca, nombre, directorio / "curva_error.png")

    umbral_val = elegir_umbral(y_val, prob_val)
    metricas_val = calcular_metricas(y_val, prob_val, umbral_val, info_val["tipo_ataque"])
    i_mejor = historial["epoca"].index(mejor_epoca) if mejor_epoca in historial["epoca"] else -1
    resumen = {
        "nombre": nombre,
        **{k: h[k] for k in ["modelo", "optimizador", "lr", "lote", "dropout", "l2", "epocas",
                             "balancear", "fuente", "fuente_test"]},
        "capas": "-".join(str(c) for c in h["capas"]) if h["modelo"] == "mlp" else "",
        "peso_real": peso_real,
        "epocas_corridas": len(historial["epoca"]),
        "mejor_epoca": mejor_epoca,
        "error_train": historial["error_train"][i_mejor] if historial["epoca"] else float("nan"),
        "error_val": historial["error_val"][i_mejor] if historial["epoca"] else float("nan"),
        "val_acer": metricas_val["acer"],
        "val_auc": metricas_val["auc"],
        "segundos": round(segundos, 1),
    }
    config.guardar_json(directorio / "hiperparametros.json",
                        {"hiperparametros": h, "huella": huella_datos(h), "resumen": resumen})

    guardar_predicciones(nombre, pd.concat([info_val, info_test], ignore_index=True),
                         np.concatenate([prob_val, prob_test]))
    return {**resumen, **resumen_test(evaluar_modelo(nombre))}


def evaluar_parametros(h, ruta_parametros, nombre, huella):
    """Evalúa un modelo ya entrenado (sus parámetros) sobre el val y el test que indica h, sin
    reentrenar. Deja la evidencia en results/experimentos/<nombre>/."""
    params = dict(np.load(ruta_parametros))
    modelo = MODELOS[h["modelo"]]
    (X_val, _, info_val), (X_test, _, info_test) = cargar_conjunto(h, "val"), cargar_conjunto(h, "test")
    directorio = config.ruta_corrida(nombre)
    directorio.mkdir(parents=True, exist_ok=True)
    config.guardar_json(directorio / "hiperparametros.json",
                        {"hiperparametros": h, "huella": huella,
                         "parametros_de": config.ruta_relativa(ruta_parametros)})
    guardar_predicciones(nombre, pd.concat([info_val, info_test], ignore_index=True),
                         np.concatenate([modelo.predecir_probabilidad(X_val, params),
                                         modelo.predecir_probabilidad(X_test, params)]))
    return evaluar_modelo(nombre)


def crear_parser():
    parser = argparse.ArgumentParser(description="Entrena y evalúa un modelo")
    for clave, (defecto, opciones) in HIPERPARAMETROS.items():
        opciones = dict(opciones)
        bandera = opciones.pop("bandera", f"--{clave}")
        parser.add_argument(bandera, dest=clave, default=defecto, **opciones)
    return parser


def main():
    resumen = ejecutar(vars(crear_parser().parse_args()))
    print(f"\nval ACER = {resumen['val_acer']:.3f}   tiempo = {resumen['segundos']} s")


if __name__ == "__main__":
    main()
