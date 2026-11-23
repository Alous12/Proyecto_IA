import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data.dataset import cargar_datos
from src.evaluate import evaluar_modelo, guardar_predicciones
from src.metrics import calcular_metricas, elegir_umbral
from src.models import mlp, regresion_logistica
from src.models.funciones_comunes import calcular_peso_real
from src.models.optimizadores import crear_optimizador

HIPERPARAMETROS_POR_DEFECTO = {
    "modelo": "regresion_logistica", "capas": [256, 64], "dropout": 0.0,
    "optimizador": "adam", "lr": 1e-3, "lote": 128, "epocas": 30, "paciencia": None,
    "l2": 0.0, "balancear": True, "fuente": "lcc", "fuente_test": None, "test_todo": False,
    "tamano": 64, "color": "gris", "semilla": config.SEMILLA, "verbose": True, "nombre": None,
}
HIPERPARAMETROS_CNN = {"epocas": 20, "paciencia": 5}


def obtener_modelo(nombre):
    if nombre == "cnn":
        from src.models import cnn
        return cnn
    return {"regresion_logistica": regresion_logistica, "mlp": mlp}[nombre]


def nombre_experimento(h):
    if h["modelo"] == "cnn":
        partes = ["cnn", str(h["tamano"]), h["color"]]
    else:
        partes = ["rl" if h["modelo"] == "regresion_logistica" else "mlp"]
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


def cargar_conjuntos(h):
    datos = {s: cargar_datos(s, h["fuente"], h["tamano"], h["color"]) for s in ["train", "val"]}
    if h["fuente_test"] == h["fuente"]:
        datos["test"] = cargar_datos("test", h["fuente"], h["tamano"], h["color"])
        return datos
    partes = [cargar_datos(s, h["fuente_test"], h["tamano"], h["color"], estadisticas_de=h["fuente"])
              for s in (config.SPLITS if h["test_todo"] else ["test"])]
    info = pd.concat([p[2] for p in partes], ignore_index=True)
    info["split"] = "test"
    datos["test"] = (np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes]), info)
    return datos


def entrenar_modelo(h, X, y, X_val, y_val, peso_real):
    if h["modelo"] == "cnn":
        return obtener_modelo("cnn").entrenar(X, y, X_val, y_val, h["epocas"], h["lote"], peso_real,
                                              h["lr"], h["paciencia"], h["semilla"], h["verbose"])
    optimizador = crear_optimizador(h["optimizador"], h["lr"])
    comunes = dict(epocas=h["epocas"], tam_lote=len(y) if optimizador.usa_lote_completo else h["lote"],
                   peso_real=peso_real, l2=h["l2"], paciencia=h["paciencia"], semilla=h["semilla"],
                   verbose=h["verbose"])
    if h["modelo"] == "mlp":
        return mlp.entrenar(X, y, X_val, y_val, optimizador, capas_ocultas=h["capas"],
                            prob_dropout=h["dropout"], **comunes)
    return regresion_logistica.entrenar(X, y, X_val, y_val, optimizador, **comunes)


def graficar_curva(historial, mejor_epoca, nombre, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historial["epoca"], historial["error_train"], label="Entrenamiento")
    ax.plot(historial["epoca"], historial["error_val"], label="Validación")
    ax.axvline(mejor_epoca, color="gray", linestyle=":", label=f"mejor época = {mejor_epoca}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error (BCE)")
    ax.set_title(f"Curva de error - {nombre}")
    ax.legend()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def ejecutar(h):
    if h.get("modelo") == "cnn":
        h = {**HIPERPARAMETROS_CNN, **h, "color": "rgb", "tamano": 64, "optimizador": "adam", "dropout": 0.3}
    h = {**HIPERPARAMETROS_POR_DEFECTO, **h}
    h["capas"] = [int(c) for c in h["capas"]]
    h["fuente_test"] = h["fuente_test"] or h["fuente"]
    nombre = h["nombre"] or nombre_experimento(h)
    directorio = config.DIR_RESULTS / nombre
    directorio.mkdir(parents=True, exist_ok=True)
    print(f"\n### {nombre}")

    datos = cargar_conjuntos(h)
    X, y, _ = datos["train"]
    X_val, y_val, info_val = datos["val"]
    X_test, _, info_test = datos["test"]
    peso_real = calcular_peso_real(y) if h["balancear"] else 1.0

    inicio = time.time()
    params, historial, mejor_epoca = entrenar_modelo(h, X, y, X_val, y_val, peso_real)
    segundos = time.time() - inicio

    modelo = obtener_modelo(h["modelo"])
    prob_val = modelo.predecir_probabilidad(X_val, params)
    prob_test = modelo.predecir_probabilidad(X_test, params)
    if h["modelo"] == "cnn":
        params.save(directorio / "modelo.keras")
    else:
        np.savez(directorio / "parametros.npz", **params)
    pd.DataFrame(historial).to_csv(directorio / "historial.csv", index=False, float_format="%.6f")
    graficar_curva(historial, mejor_epoca, nombre, directorio / "curva_error.png")

    metricas_val = calcular_metricas(y_val, prob_val, elegir_umbral(y_val, prob_val), info_val["tipo_ataque"])
    i = historial["epoca"].index(mejor_epoca) if mejor_epoca in historial["epoca"] else -1
    resumen = {
        "nombre": nombre,
        **{k: h[k] for k in ["modelo", "optimizador", "lr", "lote", "dropout", "l2", "epocas",
                             "balancear", "fuente", "fuente_test"]},
        "capas": "-".join(str(c) for c in h["capas"]) if h["modelo"] == "mlp" else "",
        "peso_real": peso_real,
        "epocas_corridas": len(historial["epoca"]),
        "mejor_epoca": mejor_epoca,
        "error_train": historial["error_train"][i] if historial["epoca"] else float("nan"),
        "error_val": historial["error_val"][i] if historial["epoca"] else float("nan"),
        "val_acer": metricas_val["acer"],
        "val_auc": metricas_val["auc"],
        "segundos": round(segundos, 1),
    }
    with open(directorio / "hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump({"hiperparametros": h, "resumen": resumen}, f, indent=2, ensure_ascii=False)

    guardar_predicciones(nombre, pd.concat([info_val, info_test], ignore_index=True),
                         np.concatenate([prob_val, prob_test]))
    metricas = evaluar_modelo(nombre)
    for nivel in ["por_frame", "por_video"]:
        for clave in ["apcer", "bpcer", "acer", "auc"]:
            resumen[f"test_{clave}_{nivel.replace('por_', '')}"] = metricas[nivel][clave]
    return resumen


def main():
    d = HIPERPARAMETROS_POR_DEFECTO
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=["regresion_logistica", "mlp", "cnn"], default=d["modelo"])
    parser.add_argument("--capas", type=int, nargs="+", default=d["capas"])
    parser.add_argument("--dropout", type=float, default=d["dropout"])
    parser.add_argument("--optimizador", choices=["gd", "minibatch", "adam"], default=d["optimizador"])
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--lote", type=int, default=d["lote"])
    parser.add_argument("--epocas", type=int, default=d["epocas"])
    parser.add_argument("--paciencia", type=int, default=d["paciencia"])
    parser.add_argument("--l2", type=float, default=d["l2"])
    parser.add_argument("--sin_balance", action="store_true")
    parser.add_argument("--fuente", choices=config.FUENTES, default=d["fuente"])
    parser.add_argument("--fuente_test", choices=config.FUENTES, default=None)
    parser.add_argument("--test_todo", action="store_true")
    parser.add_argument("--tamano", type=int, default=d["tamano"])
    parser.add_argument("--color", choices=["gris", "rgb"], default=d["color"])
    parser.add_argument("--semilla", type=int, default=d["semilla"])
    parser.add_argument("--nombre", default=None)
    parser.add_argument("--silencioso", action="store_true")
    args = parser.parse_args()
    if args.modelo == "cnn":
        parser.set_defaults(**HIPERPARAMETROS_CNN)
        args = parser.parse_args()

    h = vars(args)
    h["balancear"] = not h.pop("sin_balance")
    h["verbose"] = not h.pop("silencioso")
    resumen = ejecutar(h)
    print(f"\nval ACER = {resumen['val_acer']:.3f}   tiempo = {resumen['segundos']} s")


if __name__ == "__main__":
    main()
