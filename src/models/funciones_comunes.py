import time

import numpy as np

from src import config

EPS = 1e-7


def sigmoide(z):
    z = np.asarray(z)
    salida = np.empty_like(z, dtype=np.result_type(z.dtype, np.float32))
    positivo = z >= 0
    salida[positivo] = 1.0 / (1.0 + np.exp(-z[positivo]))
    exp_z = np.exp(z[~positivo])
    salida[~positivo] = exp_z / (1.0 + exp_z)
    return salida


def derivada_sigmoide(a):
    return a * (1.0 - a)


def relu(z):
    return np.maximum(z, 0)


def derivada_relu(z):
    return (z > 0).astype(z.dtype)


def calcular_peso_real(y):
    n_reales = np.sum(y == config.ETIQUETA_REAL)
    n_ataques = np.sum(y == config.ETIQUETA_ATAQUE)
    return float(n_ataques / n_reales) if n_reales > 0 else 1.0


def pesos_por_clase(y, peso_real=1.0):
    return np.where(y == config.ETIQUETA_REAL, peso_real, 1.0)


def binary_cross_entropy(y, p, pesos=None):
    p = np.clip(p, EPS, 1 - EPS)
    perdida = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if pesos is not None:
        perdida = perdida * pesos
    return float(np.mean(perdida))


def gradiente_bce_salida(y, p, pesos=None):
    """Derivada de la BCE (promediada) respecto al logit de la salida sigmoide: (p - y) * w / N."""
    dz = p - y
    if pesos is not None:
        dz = dz * pesos
    return (dz / len(y)).astype(p.dtype)


def penalizacion_l2(params, l2, n):
    if l2 == 0:
        return 0.0
    return float(l2 / (2 * n) * sum(np.sum(v ** 2) for k, v in params.items() if k.startswith("W")))


def copiar_parametros(params):
    return {k: v.copy() for k, v in params.items()}


def generar_lotes(n, tam_lote, generador):
    indices = generador.permutation(n)
    for inicio in range(0, n, tam_lote):
        yield indices[inicio:inicio + tam_lote]


def bucle_entrenamiento(params, paso_gradiente, funcion_error, X, y, X_val, y_val,
                        optimizador, epocas, tam_lote, peso_real=1.0, paciencia=None,
                        semilla=config.SEMILLA, verbose=True):
    """Entrena con minibatches y guarda los parámetros de la época con menor error de validación.

    paso_gradiente(X_lote, y_lote, pesos_lote, generador) -> gradientes
    funcion_error(X, y, pesos) -> error (sin dropout)
    """
    generador = np.random.default_rng(semilla)
    pesos = pesos_por_clase(y, peso_real)
    pesos_val = pesos_por_clase(y_val, peso_real)

    historial = {"epoca": [], "error_train": [], "error_val": [], "segundos": []}
    mejores, mejor_error, mejor_epoca, sin_mejora = copiar_parametros(params), np.inf, 0, 0
    inicio = time.time()

    for epoca in range(1, epocas + 1):
        for indices in generar_lotes(len(y), tam_lote, generador):
            gradientes = paso_gradiente(X[indices], y[indices], pesos[indices], generador)
            optimizador.actualizar(params, gradientes)

        error_train = funcion_error(X, y, pesos)
        error_val = funcion_error(X_val, y_val, pesos_val)
        if not (np.isfinite(error_train) and np.isfinite(error_val)):
            print(f"  [aviso] el error divergió en la época {epoca}; se detiene el entrenamiento")
            break

        historial["epoca"].append(epoca)
        historial["error_train"].append(error_train)
        historial["error_val"].append(error_val)
        historial["segundos"].append(time.time() - inicio)

        if error_val < mejor_error:
            mejores, mejor_error, mejor_epoca, sin_mejora = copiar_parametros(params), error_val, epoca, 0
        else:
            sin_mejora += 1

        if verbose:
            marca = " *" if mejor_epoca == epoca else ""
            print(f"  época {epoca:3d}  error train {error_train:.4f}  val {error_val:.4f}{marca}")
        if paciencia is not None and sin_mejora >= paciencia:
            if verbose:
                print(f"  sin mejora en {paciencia} épocas: parada temprana")
            break

    return mejores, historial, mejor_epoca
