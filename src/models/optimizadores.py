import numpy as np


class GradienteDescendente:
    def __init__(self, tasa_aprendizaje, usa_lote_completo=True):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.usa_lote_completo = usa_lote_completo

    def actualizar(self, params, gradientes):
        for k in params:
            params[k] -= self.tasa_aprendizaje * gradientes[k]


class Adam:
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


def crear_optimizador(nombre, tasa_aprendizaje):
    if nombre == "adam":
        return Adam(tasa_aprendizaje)
    return GradienteDescendente(tasa_aprendizaje, usa_lote_completo=(nombre == "gd"))
