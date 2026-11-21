"""Red de seguridad: comprueba que los resultados numéricos no cambian al limpiar el código.

    python -m tests.regresion --guardar          # guarda la referencia (antes de cambiar nada)
    python -m tests.regresion                    # compara contra la referencia
    python -m tests.regresion --figuras DIR      # además guarda las figuras sintéticas en DIR

Usa solo datos sintéticos y semilla fija, así que funciona sin los datos reales.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from src import config
from src import evaluate, experimentos, graficas
from src import verificar_gradiente as vg
from src.models import mlp, regresion_logistica
from src.models.funciones_comunes import calcular_peso_real
from src.models.optimizadores import crear_optimizador

RUTA_REFERENCIA = RAIZ / "tests" / "referencia_regresion.json"
DIR_TEMPORAL = RAIZ / "tests" / "_salida_temporal"
SEMILLA = 123


def comprobar_gradiente():
    casos = vg.casos()
    for nombre, (_, total) in casos.items():
        assert total < vg.UMBRAL_OK, f"gradiente {nombre}: error relativo {total:.2e} >= {vg.UMBRAL_OK}"
    return {nombre: total for nombre, (_, total) in casos.items()}


def datos_sinteticos(n, d, generador):
    y = (generador.random(n) < 0.75).astype(np.int64)
    X = generador.normal(0, 1, (n, d)).astype(np.float32)
    X[:, :3] += (y[:, None] * 0.8).astype(np.float32)
    return X, y


def entrenar_modelos():
    generador = np.random.default_rng(SEMILLA)
    X, y = datos_sinteticos(240, 20, generador)
    X_val, y_val = datos_sinteticos(80, 20, generador)
    peso_real = calcular_peso_real(y)

    configuraciones = {
        "rl_gd": (regresion_logistica, "gd", 0.1, {}),
        "rl_minibatch": (regresion_logistica, "minibatch", 0.05, {"l2": 1.0}),
        "rl_adam": (regresion_logistica, "adam", 0.01, {}),
        "mlp_adam": (mlp, "adam", 0.01, {"capas_ocultas": (8, 4), "prob_dropout": 0.3}),
        "mlp_minibatch": (mlp, "minibatch", 0.05, {"capas_ocultas": (6,), "l2": 0.5}),
    }
    salida, historiales = {}, {}
    for nombre, (modelo, opt, lr, extra) in configuraciones.items():
        tam_lote = len(y) if opt == "gd" else 32
        params, historial, mejor_epoca = modelo.entrenar(
            X, y, X_val, y_val, crear_optimizador(opt, lr), epocas=3, tam_lote=tam_lote,
            peso_real=peso_real, semilla=SEMILLA, verbose=False, **extra)
        salida[nombre] = {
            "error_train": historial["error_train"],
            "error_val": historial["error_val"],
            "mejor_epoca": mejor_epoca,
            "prob_val": modelo.predecir_probabilidad(X_val, params).tolist(),
        }
        historiales[nombre] = (historial, mejor_epoca)
    return salida, historiales


def predicciones_sinteticas():
    generador = np.random.default_rng(SEMILLA)
    filas = []
    for split in ["val", "test"]:
        for v in range(16):
            etiqueta = int(v % 4 != 0)
            tipo = "ninguno" if etiqueta == 0 else ("impreso" if v % 2 else "pantalla")
            base = 0.35 + 0.3 * etiqueta
            for f in range(5):
                filas.append({
                    "ruta_frame": f"sintetico/{split}_v{v}_f{f}.png", "id_video": f"{split}_v{v}",
                    "id_persona": f"{split}_p{v // 2}", "tipo_ataque": tipo, "fuente": "sintetico",
                    "split": split, "etiqueta": etiqueta,
                    "prob_ataque": float(np.clip(base + generador.normal(0, 0.18), 0, 1)),
                })
    return pd.DataFrame(filas)


def evaluar_predicciones():
    predicciones = predicciones_sinteticas()
    for nombre in ["sintetico_a", "sintetico_b"]:
        prob = predicciones["prob_ataque"] if nombre == "sintetico_a" else predicciones["prob_ataque"] ** 1.5
        evaluate.guardar_predicciones(nombre, predicciones.drop(columns="prob_ataque"), prob)
    resultado = {}
    for nombre in ["sintetico_a", "sintetico_b"]:
        metricas = evaluate.evaluar_modelo(nombre)
        for nivel in ["por_frame", "por_video"]:
            m = metricas[nivel]
            resultado[f"{nombre}/{nivel}"] = {
                **{k: m[k] for k in ["umbral", "apcer", "bpcer", "acer", "auc", "exactitud"]},
                "matriz_confusion": m["matriz_confusion"],
                "apcer_por_tipo": m["apcer_por_tipo"],
            }
    evaluate.comparar_modelos()
    resultado["comparacion_md"] = config.RUTA_COMPARACION_MD.read_text(encoding="utf-8")
    return resultado


def generar_figuras(historiales, destino):
    """Copia a `destino` las figuras de evaluate, la curva de error y una figura de grupo."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ["sintetico_a"]:
        for png in ["matriz_confusion.png", "curva_roc.png", "histograma_prob.png"]:
            shutil.copy(config.ruta_corrida(nombre, png), destino / png)

    nombres = []
    for clave, (historial, mejor_epoca) in historiales.items():
        nombre = clave.replace("_", "_sint_", 1)
        carpeta = config.ruta_corrida(nombre)
        carpeta.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(historial).drop(columns="segundos").to_csv(carpeta / "historial.csv", index=False)
        nombres.append(nombre)
    historial, mejor_epoca = historiales["mlp_adam"]
    graficas.graficar_curva(historial, mejor_epoca, "mlp_sintetico", destino / "curva_error.png")
    experimentos.graficar_grupo("grupo_sintetico", nombres)
    shutil.copy(config.DIR_FIGURAS_GRUPOS / "grupo_sintetico.png", destino / "grupo_sintetico.png")
    print(f"Figuras guardadas en {destino}")


def redirigir_results():
    """Hace que evaluate/experimentos escriban en una carpeta temporal y no en results/."""
    if DIR_TEMPORAL.exists():
        shutil.rmtree(DIR_TEMPORAL)
    DIR_TEMPORAL.mkdir(parents=True)
    config.definir_results(DIR_TEMPORAL)


def comparar(actual, referencia, ruta=""):
    errores = []
    if isinstance(referencia, dict):
        if set(actual) != set(referencia):
            return [f"{ruta}: claves distintas {sorted(actual)} vs {sorted(referencia)}"]
        for k in referencia:
            errores += comparar(actual[k], referencia[k], f"{ruta}/{k}")
    elif isinstance(referencia, str):
        if actual != referencia:
            errores.append(f"{ruta}: texto distinto")
    else:
        a, r = np.asarray(actual, dtype=float), np.asarray(referencia, dtype=float)
        if a.shape != r.shape:
            errores.append(f"{ruta}: forma {a.shape} != {r.shape}")
        elif not np.allclose(a, r, equal_nan=True):
            dif = np.abs(a - r).ravel()
            i = int(np.nanargmax(dif))
            errores.append(f"{ruta}: {int(np.sum(dif > 0))} valor(es) distintos; "
                           f"mayor diferencia en [{i}]: {a.ravel()[i]:.6g} vs {r.ravel()[i]:.6g}")
    return errores


def main():
    parser = argparse.ArgumentParser(description="Prueba de regresión con datos sintéticos")
    parser.add_argument("--guardar", action="store_true", help="guarda la referencia")
    parser.add_argument("--figuras", type=Path, default=None, help="carpeta donde guardar las figuras")
    args = parser.parse_args()

    redirigir_results()
    try:
        errores_gradiente = comprobar_gradiente()
        print("Gradiente OK:", {k: f"{v:.1e}" for k, v in errores_gradiente.items()})
        entrenamiento, historiales = entrenar_modelos()
        evaluacion = evaluar_predicciones()
        if args.figuras:
            generar_figuras(historiales, args.figuras)
    finally:
        shutil.rmtree(DIR_TEMPORAL, ignore_errors=True)

    actual = {"entrenamiento": entrenamiento, "evaluacion": evaluacion}
    if args.guardar:
        RUTA_REFERENCIA.write_text(json.dumps(actual, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nReferencia guardada en {RUTA_REFERENCIA.relative_to(RAIZ)}")
        return

    if not RUTA_REFERENCIA.exists():
        raise SystemExit("No hay referencia: corre primero con --guardar")
    referencia = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))
    errores = comparar(actual, referencia)
    if errores:
        print("\nFALLA: los resultados cambiaron respecto a la referencia:")
        for e in errores[:20]:
            print("  ", e)
        raise SystemExit(1)
    n = sum(len(v) for v in referencia.values())
    print(f"\nOK: entrenamiento ({len(entrenamiento)} modelos) y evaluación ({n - len(entrenamiento)} "
          f"bloques) coinciden con la referencia")


if __name__ == "__main__":
    main()
