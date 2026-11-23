import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.train import HIPERPARAMETROS_POR_DEFECTO, ejecutar, nombre_experimento

RUTA_CSV = config.DIR_RESULTS / "experimentos.csv"
BASE_RL = {"modelo": "regresion_logistica", "optimizador": "adam", "lr": 1e-3, "lote": 128,
           "epocas": 60, "paciencia": 15}
BASE_MLP = {**BASE_RL, "modelo": "mlp", "capas": [256, 64], "dropout": 0.3}


def con_epocas(h):
    return {**h, "epocas": 400, "paciencia": 60} if h["optimizador"] == "gd" else h


GRUPOS = {
    "optimizador": [con_epocas({**base, "optimizador": opt, "lr": lr}) for base in [BASE_RL, BASE_MLP]
                    for opt in ["gd", "minibatch", "adam"] for lr in [1e-1, 1e-2, 1e-3, 1e-4]],
    "tasa_aprendizaje": [{**base, "lr": lr} for base in [BASE_RL, BASE_MLP]
                         for lr in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]],
    "tamano_lote": [{**base, "lote": lote} for base in [BASE_RL, BASE_MLP] for lote in [32, 128, 512, 2048]],
    "arquitectura": [{**BASE_MLP, "capas": capas, "dropout": dropout}
                     for capas in [[32], [128], [256, 64], [512, 128], [512, 128, 32]]
                     for dropout in [0.0, 0.3]],
    "dropout": [{**BASE_MLP, "dropout": d} for d in [0.0, 0.2, 0.3, 0.5]],
    "regularizacion_l2": [{**base, "l2": l2} for base in [BASE_RL, BASE_MLP] for l2 in [0.0, 1.0, 10.0, 100.0]],
    "balance_clases": [{**base, "balancear": b} for base in [BASE_RL, BASE_MLP] for b in [True, False]],
}


def graficar_grupo(grupo, nombres):
    fig, ejes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, prefijo in zip(ejes, ["rl_", "mlp_"]):
        for nombre in [n for n in nombres if n.startswith(prefijo)]:
            historial = pd.read_csv(config.DIR_RESULTS / nombre / "historial.csv")
            ax.plot(historial["epoca"], historial["error_val"], label=nombre)
        if ax.lines:
            valores = pd.concat([pd.Series(l.get_ydata()) for l in ax.lines])
            ax.set_ylim(valores.min() * 0.95, min(valores.max(), valores.quantile(0.85) * 1.25))
            ax.legend(fontsize=6)
        ax.set_xlabel("Época")
        ax.set_ylabel("Error de validación")
    fig.suptitle(f"Experimentos: {grupo}")
    fig.tight_layout()
    (config.DIR_RESULTS / "experimentos").mkdir(exist_ok=True)
    fig.savefig(config.DIR_RESULTS / "experimentos" / f"{grupo}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grupo", nargs="+", choices=list(GRUPOS), default=list(GRUPOS))
    parser.add_argument("--rehacer", action="store_true")
    args = parser.parse_args()

    tabla = pd.read_csv(RUTA_CSV, keep_default_na=False) if RUTA_CSV.exists() else pd.DataFrame()
    hechos = {} if tabla.empty else {n: f for n, f in tabla.drop_duplicates("nombre").set_index("nombre").iterrows()}

    for grupo in args.grupo:
        nombres = []
        for h in GRUPOS[grupo]:
            h = {**HIPERPARAMETROS_POR_DEFECTO, "fuente_test": "lcc", **h, "verbose": False}
            nombre = nombre_experimento(h)
            if nombre in nombres:
                continue
            nombres.append(nombre)
            en_grupo = not tabla.empty and ((tabla["nombre"] == nombre) & (tabla["grupo"] == grupo)).any()
            if en_grupo and not args.rehacer:
                continue
            if nombre in hechos and not args.rehacer:
                fila = {"grupo": grupo, "nombre": nombre, **hechos[nombre].drop("grupo").to_dict()}
            else:
                fila = {"grupo": grupo, **ejecutar(h)}
                print(f"[{grupo}] {nombre}: val ACER {fila['val_acer']:.3f}, "
                      f"test ACER {fila['test_acer_frame']:.3f} ({fila['segundos']} s)")
            hechos[nombre] = pd.Series(fila).drop("nombre")
            if not tabla.empty:
                tabla = tabla[~((tabla["nombre"] == nombre) & (tabla["grupo"] == grupo))]
            tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
            tabla.sort_values(["grupo", "modelo", "val_acer"]).to_csv(RUTA_CSV, index=False, float_format="%.4f")
        graficar_grupo(grupo, nombres)

    tabla = pd.read_csv(RUTA_CSV)
    mejores = {}
    for modelo, filas in tabla.drop_duplicates("nombre").groupby("modelo"):
        mejor = filas.sort_values(["val_acer", "error_val"]).iloc[0]
        mejores[modelo] = {k: (v.item() if hasattr(v, "item") else v) for k, v in mejor.items()}
        print(f"{modelo}: {mejor['nombre']}  val ACER {mejor['val_acer']:.3f}  "
              f"test ACER {mejor['test_acer_frame']:.3f}  test AUC {mejor['test_auc_frame']:.3f}")
    with open(config.DIR_RESULTS / "mejores_modelos.json", "w", encoding="utf-8") as f:
        json.dump(mejores, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
