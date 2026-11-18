import numpy as np


def generar_lotes(n, tam_lote, generador):
    indices = generador.permutation(n)
    for inicio in range(0, n, tam_lote):
        yield indices[inicio:inicio + tam_lote]


class GradienteDescendente:
    """Descenso de gradiente por lote completo: se usa con tam_lote = N (un paso por época)."""
    nombre = "gd"
    usa_lote_completo = True

    def __init__(self, tasa_aprendizaje):
        self.tasa_aprendizaje = tasa_aprendizaje

    def actualizar(self, params, gradientes):
        for k in params:
            params[k] -= self.tasa_aprendizaje * gradientes[k]


class Minibatch(GradienteDescendente):
    """Misma regla de actualización, pero un paso por cada minibatch barajado."""
    nombre = "minibatch"
    usa_lote_completo = False


class Adam:
    nombre = "adam"
    usa_lote_completo = False

    def __init__(self, tasa_aprendizaje, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = {}, {}, 0

    def actualizar(self, params, gradientes):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            g = gradientes[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m_corregido = self.m[k] / (1 - self.beta1 ** self.t)
            v_corregido = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.tasa_aprendizaje * m_corregido / (np.sqrt(v_corregido) + self.epsilon)


OPTIMIZADORES = {"gd": GradienteDescendente, "minibatch": Minibatch, "adam": Adam}


def crear_optimizador(nombre, tasa_aprendizaje):
    if nombre not in OPTIMIZADORES:
        raise ValueError(f"Optimizador desconocido: {nombre}. Opciones: {list(OPTIMIZADORES)}")
    return OPTIMIZADORES[nombre](tasa_aprendizaje)
