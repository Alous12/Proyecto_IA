import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

from src import config
from src.metrics import agregar_por_video, elegir_umbral, metricas_desde_dataframe

COLUMNAS_PREDICCIONES = ["ruta_frame", "id_video", "id_persona", "tipo_ataque", "fuente",
                         "split", "etiqueta", "prob_ataque"]

COLOR_REAL = "#2a78d6"
COLOR_ATAQUE = "#eb6834"
COLOR_TEXTO = "#52514e"
COLOR_GUIA = "#898781"
MAPA_AZUL = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])


def guardar_predicciones(nombre_modelo, info, prob_ataque):
    predicciones = info.copy().reset_index(drop=True)
    predicciones["prob_ataque"] = np.asarray(prob_ataque, dtype=float).ravel()
    faltan = set(COLUMNAS_PREDICCIONES) - set(predicciones.columns)
    if faltan:
        raise ValueError(f"A 'info' le faltan columnas: {sorted(faltan)}")
    if not {"val", "test"} <= set(predicciones["split"]):
        raise ValueError("predicciones debe incluir frames de 'val' (umbral) y de 'test'.")

    directorio = config.DIR_RESULTS / nombre_modelo
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "predicciones.csv"
    predicciones[COLUMNAS_PREDICCIONES].to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {config.ruta_relativa(ruta)}")
    return ruta


def evaluar_modelo(nombre_modelo):
    directorio = config.DIR_RESULTS / nombre_modelo
    predicciones = pd.read_csv(directorio / "predicciones.csv", keep_default_na=False)
    val = predicciones[predicciones["split"] == "val"]
    test = predicciones[predicciones["split"] == "test"]
    val_video, test_video = agregar_por_video(val), agregar_por_video(test)

    umbral_frame = elegir_umbral(val["etiqueta"], val["prob_ataque"])
    umbral_video = elegir_umbral(val_video["etiqueta"], val_video["prob_ataque"])

    metricas = {
        "modelo": nombre_modelo,
        "fuente_val": "+".join(sorted(val["fuente"].unique())),
        "fuente_test": "+".join(sorted(test["fuente"].unique())),
        "criterio_umbral": "APCER = BPCER en validación",
        "por_frame": metricas_desde_dataframe(test, umbral_frame),
        "por_video": metricas_desde_dataframe(test_video, umbral_video),
    }
    with open(directorio / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    graficar_matrices_confusion(metricas, directorio / "matriz_confusion.png")
    graficar_roc(test, test_video, metricas, directorio / "curva_roc.png")
    graficar_histograma(test, umbral_frame, nombre_modelo, directorio / "histograma_prob.png")

    imprimir_resumen(metricas)
    return metricas


def imprimir_resumen(metricas):
    print(f"\n=== {metricas['modelo']}  (val: {metricas['fuente_val']}, "
          f"test: {metricas['fuente_test']}) ===")
    for nivel in ["por_frame", "por_video"]:
        m = metricas[nivel]
        print(f"{nivel:10s} umbral={m['umbral']:.3f}  APCER={m['apcer']:.3f}  "
              f"BPCER={m['bpcer']:.3f}  ACER={m['acer']:.3f}  "
              f"exactitud={m['exactitud']:.3f}  AUC={m['auc']:.3f}  "
              f"APCER por tipo={ {t: round(v, 3) for t, v in m['apcer_por_tipo'].items()} }")


def _estilo(ax):
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(COLOR_GUIA)
    ax.tick_params(colors=COLOR_TEXTO)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def graficar_matrices_confusion(metricas, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, nivel, titulo in zip(ejes, ["por_frame", "por_video"], ["Por frame", "Por video"]):
        matriz = np.array(metricas[nivel]["matriz_confusion"])
        porcentaje = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
        ax.imshow(porcentaje, cmap=MAPA_AZUL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if porcentaje[i, j] > 0.55 else "#0b0b0b"
                ax.text(j, i, f"{matriz[i, j]}\n({porcentaje[i, j]:.1%})",
                        ha="center", va="center", color=color, fontsize=11)
        ax.set_xticks([0, 1], ["Real", "Ataque"])
        ax.set_yticks([0, 1], ["Real", "Ataque"])
        ax.set_xlabel("Predicción", color=COLOR_TEXTO)
        ax.set_ylabel("Clase verdadera", color=COLOR_TEXTO)
        ax.set_title(f"{titulo} (umbral {metricas[nivel]['umbral']:.2f})")
        ax.tick_params(colors=COLOR_TEXTO, length=0)
        for borde in ax.spines.values():
            borde.set_visible(False)
    fig.suptitle(f"Matriz de confusión en test - {metricas['modelo']}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_roc(test, test_video, metricas, ruta):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _estilo(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_GUIA, linewidth=1)
    for df, nivel, etiqueta, color, trazo in [
            (test, "por_frame", "Por frame", COLOR_REAL, "-"),
            (test_video, "por_video", "Por video", COLOR_ATAQUE, (0, (4, 2)))]:
        if df["etiqueta"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["etiqueta"], df["prob_ataque"])
        m = metricas[nivel]
        ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=trazo,
                label=f"{etiqueta} (AUC = {m['auc']:.3f})")
        ax.plot(m["bpcer"], 1 - m["apcer"], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=2)
    ax.set_xlabel("BPCER (reales rechazados)", color=COLOR_TEXTO)
    ax.set_ylabel("1 - APCER (ataques detectados)", color=COLOR_TEXTO)
    ax.set_title(f"Curva ROC en test - {metricas['modelo']}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_histograma(test, umbral, nombre_modelo, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    _estilo(ax)
    bordes = np.linspace(0, 1, 41)
    for valor, etiqueta, color in [(config.ETIQUETA_REAL, "Real", COLOR_REAL),
                                   (config.ETIQUETA_ATAQUE, "Ataque", COLOR_ATAQUE)]:
        probs = test.loc[test["etiqueta"] == valor, "prob_ataque"]
        pesos = np.ones(len(probs)) / max(len(probs), 1)
        ax.hist(probs, bins=bordes, weights=pesos, histtype="step", linewidth=2,
                color=color, label=f"{etiqueta} (n = {len(probs)})")
    ax.axvline(umbral, color=COLOR_TEXTO, linestyle="--", linewidth=1)
    ax.annotate(f"umbral = {umbral:.2f}", (umbral, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", color=COLOR_TEXTO, va="top")
    ax.set_xlabel("Probabilidad de ataque", color=COLOR_TEXTO)
    ax.set_ylabel("Proporción de frames", color=COLOR_TEXTO)
    ax.set_title(f"Distribución de probabilidades en test (por frame) - {nombre_modelo}")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def comparar_modelos():
    filas = []
    for ruta in sorted(config.DIR_RESULTS.glob("*/metricas.json")):
        with open(ruta, encoding="utf-8") as f:
            m = json.load(f)
        for nivel in ["por_frame", "por_video"]:
            datos = m[nivel]
            fila = {
                "modelo": m["modelo"],
                "entrenamiento/val": m["fuente_val"],
                "test": m["fuente_test"],
                "nivel": nivel.replace("por_", ""),
                "APCER": datos["apcer"],
                "BPCER": datos["bpcer"],
                "ACER": datos["acer"],
                "exactitud": datos["exactitud"],
                "AUC": datos["auc"],
            }
            for tipo, valor in datos.get("apcer_por_tipo", {}).items():
                fila[f"APCER {tipo}"] = valor
            filas.append(fila)
    if not filas:
        raise SystemExit("No hay metricas.json en results/. Evalúa algún modelo primero.")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_RESULTS / "comparacion_modelos.csv", index=False, float_format="%.4f")

    columnas = list(tabla.columns)
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, fila in tabla.iterrows():
        celdas = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in fila]
        lineas.append("| " + " | ".join(celdas).replace("nan", "-") + " |")
    (config.DIR_RESULTS / "comparacion_modelos.md").write_text("\n".join(lineas) + "\n",
                                                              encoding="utf-8")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nGuardado results/comparacion_modelos.csv y results/comparacion_modelos.md")
    return tabla


def main():
    parser = argparse.ArgumentParser(description="Evalúa modelos a partir de predicciones.csv")
    parser.add_argument("--modelo", nargs="+", default=[], help="carpeta(s) dentro de results/")
    parser.add_argument("--todos", action="store_true", help="evalúa todo results/*/predicciones.csv")
    parser.add_argument("--comparar", action="store_true", help="arma la tabla comparativa")
    args = parser.parse_args()

    modelos = list(args.modelo)
    if args.todos:
        modelos = sorted(p.parent.name for p in config.DIR_RESULTS.glob("*/predicciones.csv"))
    for modelo in modelos:
        evaluar_modelo(modelo)
    if args.comparar or args.todos:
        comparar_modelos()
    if not (modelos or args.comparar):
        parser.print_help()


if __name__ == "__main__":
    main()
