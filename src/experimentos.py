"""Barrido de hiperparámetros: cambia un factor a la vez respecto a una configuración base.

    python -m src.experimentos                      # todos los grupos
    python -m src.experimentos --grupo optimizador  # solo un grupo
    python -m src.experimentos --rehacer            # vuelve a correr aunque ya estén en el CSV

Cada corrida se registra en results/experimentos.csv (y .md). Por cada grupo se guarda una
figura con las curvas de validación en results/experimentos/<grupo>.png. El mejor modelo de
cada tipo se elige por ACER en validación (no por test) y se guarda en
results/mejores_modelos.json.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.evaluate import COLOR_TEXTO, _estilo
from src.train import HIPERPARAMETROS_POR_DEFECTO, ejecutar, nombre_experimento

RUTA_CSV = config.DIR_RESULTS / "experimentos.csv"
DIR_FIGURAS = config.DIR_RESULTS / "experimentos"
PALETA = ["#2a78d6", "#eb6834", "#1f9e74", "#8e5bd1", "#c9a227", "#d6457a", "#52514e"]

BASE_RL = {"modelo": "regresion_logistica", "optimizador": "adam", "lr": 1e-3, "lote": 128,
           "epocas": 60, "paciencia": 15}
BASE_MLP = {**BASE_RL, "modelo": "mlp", "capas": [256, 64], "dropout": 0.3}
EPOCAS_GD = 400


def variantes(base, **cambios):
    return [{**base, **dict(zip(cambios, valores))} for valores in zip(*cambios.values())]


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
    return {**HIPERPARAMETROS_POR_DEFECTO, "fuente_test": "lcc", **h, "verbose": False}


def etiquetas_cortas(nombres):
    """Deja solo las partes del nombre que cambian entre las corridas del panel."""
    partes = [n.split("_") for n in nombres]
    comunes = set.intersection(*(set(p) for p in partes)) if len(partes) > 1 else set()
    return [" ".join(t for t in p if t not in comunes) or n for p, n in zip(partes, nombres)]


def graficar_grupo(grupo, nombres):
    historiales = {n: pd.read_csv(config.DIR_RESULTS / n / "historial.csv") for n in nombres
                   if (config.DIR_RESULTS / n / "historial.csv").exists()}
    paneles = [(titulo, [n for n in historiales if n.startswith(prefijo + "_")])
               for prefijo, titulo in [("rl", "Regresión logística"), ("mlp", "MLP")]]
    paneles = [(t, ns) for t, ns in paneles if ns]
    if not paneles:
        return

    fig, ejes = plt.subplots(1, len(paneles), figsize=(9 * len(paneles), 4.5), squeeze=False)
    for ax, (titulo, ns) in zip(ejes[0], paneles):
        _estilo(ax)
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
    fig.tight_layout()
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURAS / f"{grupo}.png", dpi=150)
    plt.close(fig)


def guardar_tabla(tabla):
    tabla = tabla.sort_values(["grupo", "modelo", "val_acer"])
    tabla.to_csv(RUTA_CSV, index=False, float_format="%.4f")
    columnas = ["grupo", "nombre", "mejor_epoca", "error_train", "error_val", "val_acer", "val_auc",
                "test_acer_frame", "test_auc_frame", "segundos"]
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, fila in tabla[columnas].iterrows():
        celdas = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in fila]
        lineas.append("| " + " | ".join(celdas) + " |")
    RUTA_CSV.with_suffix(".md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def elegir_mejores(tabla):
    mejores = {}
    for modelo, filas in tabla.drop_duplicates("nombre").groupby("modelo"):
        mejor = filas.sort_values(["val_acer", "error_val"]).iloc[0]
        mejores[modelo] = {k: (v.item() if hasattr(v, "item") else v) for k, v in mejor.items()}
    with open(config.DIR_RESULTS / "mejores_modelos.json", "w", encoding="utf-8") as f:
        json.dump(mejores, f, indent=2, ensure_ascii=False)
    return mejores


def main():
    parser = argparse.ArgumentParser(description="Barrido de hiperparámetros")
    parser.add_argument("--grupo", nargs="+", choices=list(GRUPOS), default=list(GRUPOS))
    parser.add_argument("--rehacer", action="store_true")
    args = parser.parse_args()

    tabla = pd.read_csv(RUTA_CSV, keep_default_na=False) if RUTA_CSV.exists() else pd.DataFrame()
    hechos = {}
    if not tabla.empty:
        hechos = {n: fila for n, fila in tabla.drop_duplicates("nombre").set_index("nombre").iterrows()}

    for grupo in args.grupo:
        nombres = []
        for h in GRUPOS[grupo]:
            h = completar(h)
            nombre = nombre_experimento(h)
            if nombre in nombres:
                continue
            nombres.append(nombre)
            ya_en_grupo = not tabla.empty and ((tabla["nombre"] == nombre) & (tabla["grupo"] == grupo)).any()
            if not args.rehacer and ya_en_grupo:
                continue
            if not args.rehacer and nombre in hechos:
                fila = {"grupo": grupo, "nombre": nombre, **hechos[nombre].drop("grupo").to_dict()}
            else:
                fila = {"grupo": grupo, **ejecutar(h)}
                print(f"[{grupo}] {nombre}: val ACER {fila['val_acer']:.3f}, "
                      f"test ACER {fila['test_acer_frame']:.3f} ({fila['segundos']} s)")
            hechos[nombre] = pd.Series(fila).drop("nombre")
            if not tabla.empty:
                tabla = tabla[~((tabla["nombre"] == nombre) & (tabla["grupo"] == grupo))]
            tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
            guardar_tabla(tabla)
        graficar_grupo(grupo, nombres)

    tabla = pd.read_csv(RUTA_CSV)
    print("\nMejor configuración de cada grupo (por ACER en validación):")
    mejores_grupo = tabla.sort_values("val_acer").groupby(["grupo", "modelo"]).head(1)
    print(mejores_grupo[["grupo", "modelo", "nombre", "val_acer", "test_acer_frame"]]
          .sort_values(["grupo", "modelo"]).to_string(index=False))

    mejores = elegir_mejores(tabla)
    print("\nModelos finales (elegidos por validación):")
    for modelo, fila in mejores.items():
        print(f"  {modelo}: {fila['nombre']}  val ACER {fila['val_acer']:.3f}  "
              f"test ACER {fila['test_acer_frame']:.3f}  test AUC {fila['test_auc_frame']:.3f}")
    print(f"\nGuardado {config.ruta_relativa(RUTA_CSV)}, experimentos.md, "
          f"experimentos/*.png y mejores_modelos.json")


if __name__ == "__main__":
    main()
