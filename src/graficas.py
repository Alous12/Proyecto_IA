"""Paleta, estilo y todas las figuras del proyecto. Los demás módulos solo llaman a estas funciones."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

from src import config

COLOR_REAL = "#2a78d6"
COLOR_ATAQUE = "#eb6834"
COLOR_TEXTO = "#52514e"
COLOR_GUIA = "#898781"
MAPA_AZUL = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
PALETA = ["#2a78d6", "#eb6834", "#1f9e74", "#8e5bd1", "#c9a227", "#d6457a", "#52514e"]


def estilo(ax):
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(COLOR_GUIA)
    ax.tick_params(colors=COLOR_TEXTO)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


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
    guardar(fig, ruta)


def graficar_roc(test, test_video, metricas, ruta):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    estilo(ax)
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
    guardar(fig, ruta)


def graficar_histograma(test, umbral, nombre_modelo, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    estilo(ax)
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
    guardar(fig, ruta)


def graficar_curva(historial, mejor_epoca, nombre, ruta):
    fig, ax = plt.subplots(figsize=(7, 4))
    estilo(ax)
    ax.plot(historial["epoca"], historial["error_train"], color=COLOR_REAL, linewidth=2,
            label="Entrenamiento")
    ax.plot(historial["epoca"], historial["error_val"], color=COLOR_ATAQUE, linewidth=2,
            linestyle=(0, (4, 2)), label="Validación")
    ax.axvline(mejor_epoca, color=COLOR_TEXTO, linestyle=":", linewidth=1)
    ax.annotate(f"mejor época = {mejor_epoca}", (mejor_epoca, ax.get_ylim()[1]),
                xytext=(6, -4), textcoords="offset points", color=COLOR_TEXTO, va="top")
    ax.set_xlabel("Época", color=COLOR_TEXTO)
    ax.set_ylabel("Error (BCE ponderada)", color=COLOR_TEXTO)
    ax.set_title(f"Curva de error - {nombre}")
    ax.legend(frameon=False)
    guardar(fig, ruta)


def etiquetas_cortas(nombres):
    """Deja solo las partes del nombre que cambian entre las corridas del panel."""
    partes = [n.split("_") for n in nombres]
    comunes = set.intersection(*(set(p) for p in partes)) if len(partes) > 1 else set()
    return [" ".join(t for t in p if t not in comunes) or n for p, n in zip(partes, nombres)]


def graficar_grupo(grupo, historiales, ruta):
    """Curvas de error de validación de un grupo de experimentos (historiales: nombre -> DataFrame)."""
    paneles = [(titulo, [n for n in historiales if n.startswith(prefijo + "_")])
               for prefijo, titulo in [("rl", "Regresión logística"), ("mlp", "MLP")]]
    paneles = [(t, ns) for t, ns in paneles if ns]
    if not paneles:
        return

    fig, ejes = plt.subplots(1, len(paneles), figsize=(9 * len(paneles), 4.5), squeeze=False)
    for ax, (titulo, ns) in zip(ejes[0], paneles):
        estilo(ax)
        valores = pd.concat([historiales[n]["error_val"] for n in ns])
        techo = min(valores.max(), valores.quantile(0.85) * 1.25)
        for i, (nombre, etiqueta) in enumerate(zip(ns, etiquetas_cortas(ns))):
            h = historiales[nombre]
            if h["error_val"].max() > techo:
                etiqueta += "  (sale del gráfico)"
            ax.plot(h["epoca"], h["error_val"], linewidth=1.6, color=PALETA[i % len(PALETA)],
                    linestyle="-" if i < len(PALETA) else (0, (4, 2)), label=etiqueta)
        ax.set_ylim(valores.min() * 0.95, techo)
        if any("_gd_" in n for n in ns):
            ax.set_xscale("log")
        ax.set_xlabel("Época", color=COLOR_TEXTO)
        ax.set_ylabel("Error de validación", color=COLOR_TEXTO)
        ax.set_title(titulo, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Experimentos: {grupo.replace('_', ' ')}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    guardar(fig, ruta)
