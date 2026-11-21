import argparse

import numpy as np
import pandas as pd

from src import config, graficas
from src.data.dataset import COLUMNAS_INFO
from src.metrics import agregar_por_video, elegir_umbral, metricas_desde_dataframe

COLUMNAS_PREDICCIONES = COLUMNAS_INFO + ["etiqueta", "prob_ataque"]


def guardar_predicciones(nombre_modelo, info, prob_ataque):
    predicciones = info.copy().reset_index(drop=True)
    predicciones["prob_ataque"] = np.asarray(prob_ataque, dtype=float).ravel()
    faltan = set(COLUMNAS_PREDICCIONES) - set(predicciones.columns)
    if faltan:
        raise ValueError(f"A 'info' le faltan columnas: {sorted(faltan)}")
    if not {"val", "test"} <= set(predicciones["split"]):
        raise ValueError("predicciones debe incluir frames de 'val' (umbral) y de 'test'.")

    directorio = config.ruta_corrida(nombre_modelo)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "predicciones.csv"
    predicciones[COLUMNAS_PREDICCIONES].to_csv(ruta, index=False)
    print(f"Predicciones guardadas en {config.ruta_relativa(ruta)}")
    return ruta


def evaluar_modelo(nombre_modelo):
    directorio = config.ruta_corrida(nombre_modelo)
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
    config.guardar_json(directorio / "metricas.json", metricas)

    graficas.graficar_matrices_confusion(metricas, directorio / "matriz_confusion.png")
    graficas.graficar_roc(test, test_video, metricas, directorio / "curva_roc.png")
    graficas.graficar_histograma(test, umbral_frame, nombre_modelo, directorio / "histograma_prob.png")

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


def tabla_markdown(tabla, nan="nan"):
    """Tabla de pandas -> texto Markdown (floats con 3 decimales; "nan" se reemplaza por `nan`)."""
    columnas = list(tabla.columns)
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, fila in tabla.iterrows():
        celdas = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in fila]
        lineas.append("| " + " | ".join(celdas).replace("nan", nan) + " |")
    return "\n".join(lineas) + "\n"


def comparar_modelos():
    filas = []
    for ruta in sorted(config.DIR_CORRIDAS.glob("*/metricas.json")):
        m = config.leer_json(ruta)
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
        raise SystemExit(f"No hay metricas.json en {config.ruta_relativa(config.DIR_CORRIDAS)}/. "
                         "Evalúa algún modelo primero.")

    tabla = pd.DataFrame(filas)
    config.DIR_RESUMEN.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(config.RUTA_COMPARACION_CSV, index=False, float_format="%.4f")

    config.RUTA_COMPARACION_MD.write_text(tabla_markdown(tabla, nan="-"), encoding="utf-8")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nGuardado {config.ruta_relativa(config.RUTA_COMPARACION_CSV)} y "
          f"{config.ruta_relativa(config.RUTA_COMPARACION_MD)}")
    return tabla


def main():
    parser = argparse.ArgumentParser(description="Evalúa modelos a partir de predicciones.csv")
    parser.add_argument("--modelo", nargs="+", default=[], help="carpeta(s) dentro de results/experimentos/")
    parser.add_argument("--todos", action="store_true", help="evalúa todo results/experimentos/*/predicciones.csv")
    parser.add_argument("--comparar", action="store_true", help="arma la tabla comparativa")
    args = parser.parse_args()

    modelos = list(args.modelo)
    if args.todos:
        modelos = sorted(p.parent.name for p in config.DIR_CORRIDAS.glob("*/predicciones.csv"))
    for modelo in modelos:
        evaluar_modelo(modelo)
    if args.comparar or args.todos:
        comparar_modelos()
    if not (modelos or args.comparar):
        parser.print_help()


if __name__ == "__main__":
    main()
