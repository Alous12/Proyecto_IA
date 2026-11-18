# Modelos y experimentos

## 1. Planteamiento

Cada rostro recortado se lleva a 64×64 en escala de grises, se aplana en un vector
$x \in \mathbb{R}^{4096}$ y se normaliza píxel a píxel con la media y la desviación de train
(sección de datos). La tarea es binaria: $y = 0$ real, $y = 1$ ataque. Ambos modelos devuelven
$p = P(y = 1 \mid x)$, la probabilidad de ataque.

Los dos modelos y sus optimizadores están programados desde cero con numpy
(`src/models/`). Se entrena con minibatches de $m$ ejemplos; $X \in \mathbb{R}^{m \times 4096}$
tiene un ejemplo por fila.

### Función de error: BCE ponderada

LCC FASD está muy desbalanceado (85 % de ataques en train). Con la BCE normal, el modelo puede
reducir el error diciendo casi siempre "ataque". Por eso cada ejemplo real se pondera con
$w_{real} = n_{ataques} / n_{reales} \approx 5.8$ (calculado en train) y cada ataque con
$w = 1$:

$$
J = -\frac{1}{m}\sum_{i=1}^{m} w_i\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
    + \frac{\lambda}{2m}\sum_l \lVert W^{[l]}\rVert^2
$$

El segundo término es la regularización L2 opcional ($\lambda = 0$ por defecto). Con salida
sigmoide, la derivada respecto al logit de salida $z$ se simplifica a

$$
\frac{\partial J}{\partial z_i} = \frac{w_i\,(p_i - y_i)}{m}
$$

que usan los dos modelos (`gradiente_bce_salida` en `funciones_comunes.py`). Para evitar
$\log 0$, $p$ se recorta a $[10^{-7}, 1-10^{-7}]$. La sigmoide se calcula de forma estable
($e^{z}/(1+e^{z})$ cuando $z < 0$) para que no haya overflow.

### Entrenamiento y selección

En cada época se barajan los ejemplos de train, se recorren en minibatches y, al final, se
calcula el error (sin dropout) en train y en validación. Se guardan los parámetros de la
**época con menor error de validación** y, si se indica `paciencia`, se detiene el
entrenamiento cuando el error de validación no mejora en esa cantidad de épocas.

El umbral de decisión no es 0.5: `src.evaluate` lo elige en validación donde APCER = BPCER y
lo aplica en test.

## 2. Regresión logística (baseline)

**Arquitectura:** una sola neurona con sigmoide sobre los 4096 píxeles (4097 parámetros).

**Forward:**

$$
z = XW + b,\qquad p = \sigma(z) = \frac{1}{1+e^{-z}}
$$

**Gradiente:**

$$
dz = \frac{w \odot (p - y)}{m},\qquad
\frac{\partial J}{\partial W} = X^\top dz + \frac{\lambda}{m}W,\qquad
\frac{\partial J}{\partial b} = \sum_i dz_i
$$

**Inicialización:** $W \sim \mathcal{N}(0, 0.01^2)$, $b = 0$.

Funciones en `regresion_logistica.py`: `inicializar_parametros`, `calcular_logits`,
`calcular_error`, `calcular_gradiente`, `actualizar_parametros`, `entrenar` y
`predecir_probabilidad`.

## 3. Red feed-forward (MLP)

**Arquitectura base:** 4096 → 256 → 64 → 1, con ReLU en las capas ocultas, dropout en las
capas ocultas y sigmoide en la salida (unos 1.07 M parámetros).

**Forward** (capa $l = 1 \dots L$, $A^{[0]} = X$):

$$
Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}
$$

$$
A^{[l]} = \mathrm{ReLU}(Z^{[l]}) \odot \frac{D^{[l]}}{1-q}\quad (l < L),\qquad
A^{[L]} = \sigma(Z^{[L]}) = p
$$

$D^{[l]}$ es una máscara de Bernoulli que apaga cada neurona oculta con probabilidad $q$
(*inverted dropout*). Al dividir entre $1-q$ en el entrenamiento, al predecir basta con
apagar el dropout sin reescalar. En `propagacion_adelante` se guardan $Z^{[l]}$, $A^{[l]}$ y
$D^{[l]}$ de cada capa para usarlos en backprop.

**Backpropagation:**

$$
dZ^{[L]} = \frac{w \odot (A^{[L]} - y)}{m}
$$

$$
dW^{[l]} = A^{[l-1]\top} dZ^{[l]} + \frac{\lambda}{m}W^{[l]},\qquad
db^{[l]} = \sum_i dZ^{[l]}_{i,:}
$$

$$
dA^{[l-1]} = dZ^{[l]}W^{[l]\top} \odot \frac{D^{[l-1]}}{1-q},\qquad
dZ^{[l-1]} = dA^{[l-1]} \odot \mathbb{1}[Z^{[l-1]} > 0]
$$

**Inicialización He:** $W^{[l]} \sim \mathcal{N}\!\left(0, \tfrac{2}{n_{l-1}}\right)$,
$b^{[l]} = 0$. Es la adecuada para ReLU, porque mantiene la varianza de las activaciones
entre capas.

Funciones en `mlp.py`: `inicializar_parametros` (por capas), `propagacion_adelante`,
`calcular_error`, `backpropagation`, `entrenar` y `predecir_probabilidad`.

## 4. Optimizadores

Están en `optimizadores.py` y comparten la interfaz `actualizar(parametros, gradientes)`.

| Optimizador | Regla | Pasos por época |
|---|---|---|
| Gradient descent | $\theta \leftarrow \theta - \alpha\,\nabla J$ con todo train | 1 |
| Minibatch | la misma regla, con un minibatch barajado | $\lceil N/m \rceil$ |
| Adam | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2$, $\theta \leftarrow \theta - \alpha\,\hat m_t / (\sqrt{\hat v_t}+\epsilon)$ con $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ | $\lceil N/m \rceil$ |

Adam usa $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\epsilon = 10^{-8}$. Como gradient descent da un
solo paso por época, en los experimentos se le dan 400 épocas, contra 60 de los demás.

## 5. Verificación del gradiente

`python -m src.verificar_gradiente` compara el gradiente de backprop con uno numérico por
diferencias centrales:

$$
\frac{\partial J}{\partial \theta_j} \approx \frac{J(\theta + \epsilon e_j) - J(\theta - \epsilon e_j)}{2\epsilon},
\qquad \epsilon = 10^{-5}
$$

La prueba usa 12 ejemplos con 6 entradas, en float64. El criterio es el error relativo
$\lVert g_{bp} - g_{num}\rVert / (\lVert g_{bp}\rVert + \lVert g_{num}\rVert) < 10^{-7}$.

| Caso | Error relativo |
|---|---|
| Regresión logística | 1.9e-11 |
| Regresión logística, L2 = 0.1, $w_{real}$ = 2 | 1.8e-11 |
| MLP 6-5-1 | 3.1e-11 |
| MLP 6-5-4-1, L2 = 0.1, $w_{real}$ = 2 | 6.7e-11 |
| MLP 6-5-4-1, dropout 0.3, L2 = 0.1, $w_{real}$ = 2 | 4.8e-11 |

Todos los casos quedan unos cuatro órdenes de magnitud por debajo del umbral, así que las
derivadas son correctas. La salida completa por parámetro está en
`results/verificacion_gradiente.txt`.

Dos detalles de la prueba:
- Con dropout, la máscara se genera siempre con la misma semilla, para que $J$ sea una
  función determinista de $\theta$.
- Los sesgos se inicializan con valores pequeños positivos. Si el dropout apaga todas las
  neuronas de una capa para un ejemplo, $Z = b = 0$ cae justo en el punto donde ReLU no es
  derivable, y la comparación numérica deja de ser válida ahí. Pasó en la primera versión de
  la prueba: $b^{[2]}$ daba un error de 8.5e-2.

## 6. Experimentos

**Datos:** LCC FASD, 64×64 en gris. Train tiene 9456 frames de 111 personas, validación
1791 frames de 20 personas y test (evaluation oficial) 7580 frames. Hay 85–96 % de ataques
según el split.

**Protocolo** (`python -m src.experimentos`):
- Se parte de una configuración base y se cambia **un factor a la vez**.
  - Base de la regresión logística: Adam, $\alpha = 10^{-3}$, lote 128, BCE ponderada.
  - Base del MLP: lo mismo, más capas 256-64 y dropout 0.3.
- Gradient descent tiene un máximo de 400 épocas y paciencia 60. Los demás, 60 épocas y
  paciencia 15.
- Cada corrida se queda con la época de menor error de validación.
- La configuración se compara por **ACER en validación**, con el umbral APCER = BPCER elegido
  también en validación. El test se reporta, pero no se usa para elegir.
- Semilla fija (42).

En total hubo 53 configuraciones distintas, todas en CPU (Apple M5). Cada una tarda entre
0.1 y 25 s.

La tabla completa está en `results/experimentos.csv` y `results/experimentos.md`. Las curvas
de validación de cada grupo están en `results/experimentos/<grupo>.png`.

### 6.1 Resultados por factor

ACER en validación (menor es mejor). La fila marcada como "base" es la configuración de
partida.

**Optimizador y tasa de aprendizaje**

| Optimizador | α | RL val ACER | MLP val ACER |
|---|---|---|---|
| GD (lote completo) | 1e-1 | 0.332 (diverge, error ≈ 9) | 0.290 |
| GD | 1e-2 | 0.401 | 0.276 (mejor época 188) |
| GD | 1e-3 | **0.324** (mejor época 348) | 0.321 (no converge en 400) |
| GD | 1e-4 | 0.385 | 0.427 |
| Minibatch | 1e-1 | 0.354 (inestable) | 0.312 |
| Minibatch | 1e-2 | 0.372 | **0.269** |
| Minibatch | 1e-3 | 0.328 | 0.280 |
| Minibatch | 1e-4 | **0.324** | 0.325 |
| Adam | 1e-1 | 0.376 (inestable) | 0.500 (colapsa: AUC = 0.5) |
| Adam | 1e-2 | 0.377 | 0.375 |
| Adam | 3e-3 | 0.376 | **0.261** |
| Adam | 1e-3 (base) | 0.361 | 0.313 |
| Adam | 3e-4 | 0.341 | 0.274 |
| Adam | 1e-4 | 0.329 | 0.282 |

**Tamaño de lote** (Adam, α = 1e-3)

| Lote | RL | MLP |
|---|---|---|
| 32 | 0.357 | 0.300 |
| 128 (base) | 0.361 | 0.313 |
| 512 | 0.355 | **0.273** |
| 2048 | **0.327** | 0.316 |

**Arquitectura y dropout del MLP**

| Capas ocultas | Sin dropout | Dropout 0.3 |
|---|---|---|
| 32 | 0.283 | 0.309 |
| 128 | 0.309 | 0.317 |
| 256-64 | **0.280** | 0.313 |
| 512-128 | 0.297 | 0.301 |
| 512-128-32 | 0.295 | 0.284 |

| Dropout (256-64) | 0 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|
| val ACER | 0.280 | **0.276** | 0.313 | 0.304 |
| error val (BCE) | 1.548 | 1.069 | 1.036 | **0.936** |

**Regularización L2 y balance de clases**

| L2 (λ) | 0 (base) | 1 | 10 | 100 |
|---|---|---|---|---|
| RL | 0.361 | 0.361 | 0.362 | **0.319** |
| MLP | 0.313 | **0.275** | 0.287 | 0.360 |

| BCE | RL | MLP |
|---|---|---|
| Ponderada (base) | **0.361** | 0.313 |
| Sin ponderar | 0.401 | **0.280** |

### 6.2 Análisis

1. **El sobreajuste domina.** En casi todas las corridas el error de train sigue bajando
   mientras el de validación llega a su mínimo en las primeras 1–9 épocas y luego sube (ver
   `results/experimentos/arquitectura.png`). Con píxeles crudos y solo 111 personas en train,
   el modelo aprende detalles de las personas y de las condiciones de captura, no la textura
   del ataque. Por eso guardar la mejor época según validación es imprescindible.
2. **Optimizadores.**
   - Gradient descent da curvas suaves, pero necesita cientos de épocas: el MLP con α = 1e-2
     llega a su mejor época en la 188, y con α = 1e-3 no converge en 400.
   - Minibatch y Adam llegan a un error similar o menor en menos de 10 épocas, aunque con
     curvas más ruidosas.
   - Con α = 0.1, la regresión logística diverge en GD y en minibatch, y el MLP con Adam
     colapsa: predice una constante (AUC 0.5), probablemente porque las ReLU se quedan
     "muertas".
   - Adam funciona mejor con α entre 1e-4 y 3e-3.
3. **Tamaño de lote.** Con la regresión logística, el lote grande (2048) da el mejor
   resultado. El gradiente es menos ruidoso y los pasos se parecen a los de GD, que también
   fue el mejor para RL. En el MLP, 512 es el mejor.
4. **Capacidad y regularización.**
   - La arquitectura importa poco: entre 32 y 512-128-32 neuronas, el ACER va de 0.28 a 0.32.
   - El dropout baja siempre el **error** de validación (por ejemplo, de 1.55 a 0.94 con
     q = 0.5), porque evita que el modelo se vuelva sobreconfiado. Sin embargo, no siempre
     mejora el ACER, que solo depende del orden de las probabilidades y del umbral.
   - L2 ayuda a los dos modelos: λ = 100 en la RL (0.361 → 0.319) y λ = 1 en el MLP
     (0.313 → 0.275). Un L2 demasiado fuerte empeora el MLP.
5. **Balance de clases.**
   - Ponderar la BCE mejora la regresión logística (0.401 → 0.361).
   - En el MLP, la versión sin ponderar sale algo mejor en ACER. Como el umbral se ajusta en
     validación para igualar APCER y BPCER, ese ajuste compensa buena parte del desbalance; el
     peso cambia sobre todo la calibración de las probabilidades.
6. **Precisión de las comparaciones.** Validación tiene solo 20 personas, así que diferencias
   de ACER menores a unos 0.03 están dentro del ruido. Por ejemplo, dropout 0.5 es peor que
   0.2 en validación, pero mejor en test. Las tendencias claras son estas: el MLP supera a la
   RL; un α grande es perjudicial; GD necesita muchas más épocas; y regularizar ayuda.

### 6.3 Modelos finales

Se eligió la mejor configuración de cada modelo por ACER en validación
(`results/mejores_modelos.json`). Las métricas de test son por frame; en LCC, por frame y por
video coinciden, porque cada imagen de test es su propio "video".

| Modelo | Configuración | val ACER | test APCER | test BPCER | test ACER | test AUC |
|---|---|---|---|---|---|---|
| Regresión logística | Adam, α = 1e-3, lote 128, L2 = 100 | 0.319 | 0.406 | 0.290 | 0.348 | 0.693 |
| MLP | 256-64, dropout 0.3, Adam, α = 3e-3, lote 128 | 0.261 | 0.333 | 0.245 | **0.289** | **0.787** |

El MLP reduce el ACER de test en unos 6 puntos y sube el AUC de 0.69 a 0.79 respecto al
baseline. Las capas ocultas permiten combinar los píxeles de forma no lineal, y con eso probablemente
captan patrones de textura (moiré de la pantalla, brillo, nitidez) que una frontera lineal no
separa.

Aun así, el MLP deja pasar 1 de cada 3 ataques con un BPCER del 25 %. Para un sistema real
todavía es alto.

### 6.4 Limitaciones y trabajo posterior

- Una sola semilla. Conviene repetir los mejores modelos con 3–5 semillas y reportar media y
  desviación.
- Se varió un factor a la vez. Combinar los mejores, por ejemplo minibatch con α = 1e-2 y
  L2 = 1, podría mejorar el resultado, pero debe decidirse con validación.
- Falta la prueba cruzada con el dataset propio (`--fuente_test propio --test_todo`). Mediría
  cuánto generalizan los modelos a otra cámara y a otras personas.
- Los píxeles crudos de 64×64 en gris pierden información de color y de alta frecuencia. Un
  descriptor de textura como LBP o los recortes en RGB son la mejora más directa.
