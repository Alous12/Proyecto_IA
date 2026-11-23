import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src import config
from src.metrics import agregar_por_video, calcular_metricas, elegir_umbral

COLUMNAS_PREDICCIONES = ["ruta_frame", "id_video", "id_persona", "tipo_ataque", "fuente",
                         "split", "etiqueta", "prob_ataque"]


def guardar_predicciones(nombre_modelo, info, prob_ataque):
    predicciones = info.copy().reset_index(drop=True)
    predicciones["prob_ataque"] = np.asarray(prob_ataque, dtype=float).ravel()
    directorio = config.DIR_RESULTS / nombre_modelo
    directorio.mkdir(parents=True, exist_ok=True)
    predicciones[COLUMNAS_PREDICCIONES].to_csv(directorio / "predicciones.csv", index=False)


def evaluar_modelo(nombre_modelo):
    directorio = config.DIR_RESULTS / nombre_modelo
    predicciones = pd.read_csv(directorio / "predicciones.csv", keep_default_na=False)
    niveles = {"por_frame": predicciones, "por_video": agregar_por_video(predicciones)}

    metricas = {"modelo": nombre_modelo,
                "fuente_val": "+".join(sorted(predicciones.loc[predicciones["split"] == "val", "fuente"].unique())),
                "fuente_test": "+".join(sorted(predicciones.loc[predicciones["split"] == "test", "fuente"].unique())),
                "criterio_umbral": "APCER = BPCER en validación"}
    tests = {}
    for nivel, df in niveles.items():
        val, test = df[df["split"] == "val"], df[df["split"] == "test"]
        umbral = elegir_umbral(val["etiqueta"], val["prob_ataque"])
        metricas[nivel] = calcular_metricas(test["etiqueta"], test["prob_ataque"], umbral, test["tipo_ataque"])
        tests[nivel] = test
    with open(directorio / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    graficar(metricas, tests, directorio)
    imprimir_resumen(metricas)
    return metricas


def imprimir_resumen(metricas):
    print(f"\n=== {metricas['modelo']}  (val: {metricas['fuente_val']}, test: {metricas['fuente_test']}) ===")
    for nivel in ["por_frame", "por_video"]:
        m = metricas[nivel]
        print(f"{nivel:10s} umbral={m['umbral']:.3f}  APCER={m['apcer']:.3f}  BPCER={m['bpcer']:.3f}  "
              f"ACER={m['acer']:.3f}  exactitud={m['exactitud']:.3f}  AUC={m['auc']:.3f}")


def graficar(metricas, tests, directorio):
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, nivel in zip(ejes, ["por_frame", "por_video"]):
        matriz = np.array(metricas[nivel]["matriz_confusion"])
        ax.imshow(matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, matriz[i, j], ha="center", va="center")
        ax.set_xticks([0, 1], ["Real", "Ataque"])
        ax.set_yticks([0, 1], ["Real", "Ataque"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase verdadera")
        ax.set_title(f"{nivel} (umbral {metricas[nivel]['umbral']:.2f})")
    fig.tight_layout()
    fig.savefig(directorio / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for nivel, test in tests.items():
        if test["etiqueta"].nunique() == 2:
            fpr, tpr, _ = roc_curve(test["etiqueta"], test["prob_ataque"])
            ax.plot(fpr, tpr, label=f"{nivel} (AUC = {metricas[nivel]['auc']:.3f})")
    ax.set_xlabel("BPCER")
    ax.set_ylabel("1 - APCER")
    ax.set_title(f"Curva ROC - {metricas['modelo']}")
    ax.legend()
    fig.savefig(directorio / "curva_roc.png", dpi=150)
    plt.close(fig)

    test = tests["por_frame"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for valor, etiqueta in [(config.ETIQUETA_REAL, "Real"), (config.ETIQUETA_ATAQUE, "Ataque")]:
        ax.hist(test.loc[test["etiqueta"] == valor, "prob_ataque"], bins=np.linspace(0, 1, 41),
                density=True, histtype="step", label=etiqueta)
    ax.axvline(metricas["por_frame"]["umbral"], color="gray", linestyle="--")
    ax.set_xlabel("Probabilidad de ataque")
    ax.set_title(f"Probabilidades en test - {metricas['modelo']}")
    ax.legend()
    fig.savefig(directorio / "histograma_prob.png", dpi=150)
    plt.close(fig)


def comparar_modelos():
    filas = []
    for ruta in sorted(config.DIR_RESULTS.glob("*/metricas.json")):
        with open(ruta, encoding="utf-8") as f:
            m = json.load(f)
        for nivel in ["por_frame", "por_video"]:
            filas.append({"modelo": m["modelo"], "entrenamiento/val": m["fuente_val"], "test": m["fuente_test"],
                          "nivel": nivel.replace("por_", ""),
                          **{k.upper() if k != "exactitud" else k: m[nivel][k]
                             for k in ["apcer", "bpcer", "acer", "exactitud", "auc"]}})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_RESULTS / "comparacion_modelos.csv", index=False, float_format="%.4f")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return tabla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", nargs="+", default=[])
    parser.add_argument("--todos", action="store_true")
    args = parser.parse_args()
    modelos = args.modelo
    if args.todos:
        modelos = sorted(p.parent.name for p in config.DIR_RESULTS.glob("*/predicciones.csv"))
    for modelo in modelos:
        evaluar_modelo(modelo)
    if args.todos:
        comparar_modelos()


if __name__ == "__main__":
    main()
