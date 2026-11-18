"""Compara el gradiente analítico (backprop) con uno numérico por diferencias centrales.

    python -m src.verificar_gradiente

Se usa un ejemplo pequeño en float64. Un error relativo < 1e-7 indica que las derivadas
están bien; > 1e-4 indica casi seguro un error. El resultado se guarda en
results/verificacion_gradiente.txt como evidencia para el documento.
"""
import numpy as np

from src import config
from src.models import mlp, regresion_logistica
from src.models.funciones_comunes import binary_cross_entropy, penalizacion_l2, pesos_por_clase

EPSILON = 1e-5
UMBRAL_OK = 1e-7


def gradiente_numerico(funcion_error, params, epsilon=EPSILON):
    numerico = {}
    for k, valor in params.items():
        grad = np.zeros_like(valor)
        for i in np.ndindex(valor.shape):
            original = valor[i]
            valor[i] = original + epsilon
            error_mas = funcion_error()
            valor[i] = original - epsilon
            error_menos = funcion_error()
            valor[i] = original
            grad[i] = (error_mas - error_menos) / (2 * epsilon)
        numerico[k] = grad
    return numerico


def error_relativo(a, b):
    a, b = np.ravel(a), np.ravel(b)
    denominador = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denominador) if denominador > 0 else 0.0


def comparar(nombre, analitico, numerico):
    lineas = [f"\n{nombre}", f"  {'parámetro':10s} {'forma':>10s} {'error relativo':>16s}"]
    for k in analitico:
        lineas.append(f"  {k:10s} {str(analitico[k].shape):>10s} "
                      f"{error_relativo(analitico[k], numerico[k]):16.2e}")
    total = error_relativo(np.concatenate([analitico[k].ravel() for k in analitico]),
                           np.concatenate([numerico[k].ravel() for k in analitico]))
    estado = "OK" if total < UMBRAL_OK else "REVISAR"
    lineas.append(f"  {'total':10s} {'':>10s} {total:16.2e}   -> {estado}")
    return lineas, total


def datos_de_prueba(n=12, d=6, semilla=config.SEMILLA):
    generador = np.random.default_rng(semilla)
    X = generador.normal(0, 1, (n, d))
    y = (generador.random(n) < 0.6).astype(float)
    y[:2] = [0, 1]
    return X, y


def verificar_regresion_logistica(X, y, pesos, l2):
    params = regresion_logistica.inicializar_parametros(X.shape[1], dtype=np.float64)
    params["W"] = np.random.default_rng(1).normal(0, 0.5, X.shape[1])
    params["b"] = np.array([0.1])
    analitico = regresion_logistica.calcular_gradiente(X, y, params, pesos, l2)
    numerico = gradiente_numerico(
        lambda: regresion_logistica.calcular_error(X, y, params, pesos, l2), params)
    return comparar(f"Regresión logística (l2 = {l2}, peso_real = {pesos.max():.1f})",
                    analitico, numerico)


def verificar_mlp(X, y, pesos, l2, capas_ocultas, prob_dropout):
    params = mlp.inicializar_parametros([X.shape[1], *capas_ocultas, 1], dtype=np.float64)
    # Sesgos no nulos: si el dropout apaga toda una capa en un ejemplo, Z = b = 0 cae justo en
    # el quiebre de ReLU (no derivable) y la comparación numérica no es válida en ese punto.
    generador = np.random.default_rng(3)
    for k in params:
        if k.startswith("b"):
            params[k] = generador.uniform(0.05, 0.2, params[k].shape)

    def forward():
        # Misma semilla en cada llamada -> misma máscara de dropout -> función determinista
        return mlp.propagacion_adelante(X, params, prob_dropout, True, np.random.default_rng(7))

    def error():
        p, _ = forward()
        return binary_cross_entropy(y, p, pesos) + penalizacion_l2(params, l2, len(y))

    _, cache = forward()
    analitico = mlp.backpropagation(cache, y, params, pesos, l2)
    numerico = gradiente_numerico(error, params)
    capas = "-".join(str(c) for c in [X.shape[1], *capas_ocultas, 1])
    return comparar(f"MLP {capas} (dropout = {prob_dropout}, l2 = {l2}, "
                    f"peso_real = {pesos.max():.1f})", analitico, numerico)


def main():
    X, y = datos_de_prueba()
    pesos = pesos_por_clase(y, peso_real=2.0)

    casos = [
        verificar_regresion_logistica(X, y, np.ones_like(y), 0.0),
        verificar_regresion_logistica(X, y, pesos, 0.1),
        verificar_mlp(X, y, np.ones_like(y), 0.0, [5], 0.0),
        verificar_mlp(X, y, pesos, 0.1, [5, 4], 0.0),
        verificar_mlp(X, y, pesos, 0.1, [5, 4], 0.3),
    ]

    encabezado = [f"Verificación de gradiente: diferencias centrales, epsilon = {EPSILON}, "
                  f"{X.shape[0]} ejemplos con {X.shape[1]} entradas, float64.",
                  f"Error relativo = ||g_backprop - g_numerico|| / (||g_backprop|| + ||g_numerico||); "
                  f"OK si < {UMBRAL_OK:g}."]
    lineas = encabezado + [l for caso, _ in casos for l in caso]
    texto = "\n".join(lineas)
    print(texto)

    config.DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    ruta = config.DIR_RESULTS / "verificacion_gradiente.txt"
    ruta.write_text(texto + "\n", encoding="utf-8")
    print(f"\nGuardado {config.ruta_relativa(ruta)}")

    if any(total >= UMBRAL_OK for _, total in casos):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
