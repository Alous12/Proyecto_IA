"""Regresión logística (baseline) en numpy.

p = sigmoide(X W + b),   J = BCE ponderada + l2 / (2N) * ||W||^2
"""
import numpy as np

from src import config
from src.models.funciones_comunes import (binary_cross_entropy, bucle_entrenamiento,
                                          gradiente_bce_salida, penalizacion_l2, sigmoide)


def inicializar_parametros(n_entradas, semilla=config.SEMILLA, dtype=np.float32):
    generador = np.random.default_rng(semilla)
    return {
        "W": (generador.normal(0, 0.01, n_entradas)).astype(dtype),
        "b": np.zeros(1, dtype=dtype),
    }


def calcular_logits(X, params):
    return X @ params["W"] + params["b"][0]


def predecir_probabilidad(X, params):
    return sigmoide(calcular_logits(X, params))


def calcular_error(X, y, params, pesos=None, l2=0.0):
    p = predecir_probabilidad(X, params)
    return binary_cross_entropy(y, p, pesos) + penalizacion_l2(params, l2, len(y))


def calcular_gradiente(X, y, params, pesos=None, l2=0.0):
    p = predecir_probabilidad(X, params)
    dz = gradiente_bce_salida(y, p, pesos)
    dW = X.T @ dz
    if l2:
        dW = dW + (l2 / len(y)) * params["W"]
    return {"W": dW.astype(params["W"].dtype), "b": np.array([dz.sum()], dtype=params["b"].dtype)}


def actualizar_parametros(params, gradientes, optimizador):
    optimizador.actualizar(params, gradientes)
    return params


def entrenar(X, y, X_val, y_val, optimizador, epocas=30, tam_lote=128, peso_real=1.0,
             l2=0.0, paciencia=None, semilla=config.SEMILLA, verbose=True):
    params = inicializar_parametros(X.shape[1], semilla, dtype=X.dtype)

    def paso_gradiente(X_lote, y_lote, pesos_lote, _generador):
        return calcular_gradiente(X_lote, y_lote, params, pesos_lote, l2)

    def funcion_error(X_e, y_e, pesos_e):
        return calcular_error(X_e, y_e, params, pesos_e, l2)

    return bucle_entrenamiento(params, paso_gradiente, funcion_error, X, y, X_val, y_val,
                               optimizador, epocas, tam_lote, peso_real, paciencia, semilla, verbose)
