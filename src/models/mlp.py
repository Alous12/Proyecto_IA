import numpy as np

from src import config
from src.models.funciones_comunes import (binary_cross_entropy, bucle_entrenamiento,
                                          derivada_relu, gradiente_bce_salida,
                                          penalizacion_l2, relu, sigmoide)


def inicializar_parametros(capas, semilla=config.SEMILLA, dtype=np.float32):
    generador = np.random.default_rng(semilla)
    params = {}
    for l in range(1, len(capas)):
        escala = np.sqrt(2.0 / capas[l - 1])
        params[f"W{l}"] = (generador.normal(0, 1, (capas[l - 1], capas[l])) * escala).astype(dtype)
        params[f"b{l}"] = np.zeros(capas[l], dtype=dtype)
    return params


def propagacion_adelante(X, params, prob_dropout=0.0, generador=None):
    L = len(params) // 2
    cache = {"A0": X}
    A = X
    for l in range(1, L + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        if l < L:
            A = relu(Z)
            if generador is not None and prob_dropout > 0:
                mascara = (generador.random(A.shape) >= prob_dropout).astype(A.dtype) / (1 - prob_dropout)
                A = A * mascara
                cache[f"D{l}"] = mascara
        else:
            A = sigmoide(Z)
        cache[f"Z{l}"], cache[f"A{l}"] = Z, A
    return A[:, 0], cache


def backpropagation(cache, y, params, pesos=None, l2=0.0):
    L = len(params) // 2
    dZ = gradiente_bce_salida(y, cache[f"A{L}"][:, 0], pesos)[:, None]
    gradientes = {}
    for l in range(L, 0, -1):
        W = params[f"W{l}"]
        dW = cache[f"A{l - 1}"].T @ dZ
        if l2:
            dW = dW + (l2 / len(y)) * W
        gradientes[f"W{l}"] = dW.astype(W.dtype)
        gradientes[f"b{l}"] = dZ.sum(axis=0).astype(W.dtype)
        if l > 1:
            dA = dZ @ W.T
            if f"D{l - 1}" in cache:
                dA = dA * cache[f"D{l - 1}"]
            dZ = dA * derivada_relu(cache[f"Z{l - 1}"])
    return gradientes


def predecir_probabilidad(X, params, tam_lote=4096):
    return np.concatenate([propagacion_adelante(X[i:i + tam_lote], params)[0]
                           for i in range(0, len(X), tam_lote)])


def entrenar(X, y, X_val, y_val, optimizador, capas_ocultas=(256, 64), prob_dropout=0.0,
             epocas=30, tam_lote=128, peso_real=1.0, l2=0.0, paciencia=None,
             semilla=config.SEMILLA, verbose=True):
    params = inicializar_parametros([X.shape[1], *capas_ocultas, 1], semilla, dtype=X.dtype)

    def paso_gradiente(X_lote, y_lote, pesos_lote, generador):
        _, cache = propagacion_adelante(X_lote, params, prob_dropout, generador)
        return backpropagation(cache, y_lote, params, pesos_lote, l2)

    def funcion_error(X_e, y_e, pesos_e):
        p = predecir_probabilidad(X_e, params)
        return binary_cross_entropy(y_e, p, pesos_e) + penalizacion_l2(params, l2, len(y_e))

    return bucle_entrenamiento(params, paso_gradiente, funcion_error, X, y, X_val, y_val,
                               optimizador, epocas, tam_lote, peso_real, paciencia, semilla, verbose)
