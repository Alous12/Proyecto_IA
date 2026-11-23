import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src import config


def elegir_umbral(etiquetas, prob_ataque):
    fpr, tpr, umbrales = roc_curve(etiquetas, prob_ataque)
    return float(min(umbrales[int(np.argmin(np.abs((1 - tpr) - fpr)))], 1.0))


def calcular_metricas(etiquetas, prob_ataque, umbral, tipos_ataque):
    etiquetas = np.asarray(etiquetas).astype(int)
    prob_ataque = np.asarray(prob_ataque, dtype=float)
    tipos_ataque = np.asarray(tipos_ataque)
    prediccion = (prob_ataque >= umbral).astype(int)
    es_ataque = etiquetas == config.ETIQUETA_ATAQUE
    es_real = etiquetas == config.ETIQUETA_REAL

    vn = int(np.sum(es_real & (prediccion == 0)))
    fp = int(np.sum(es_real & (prediccion == 1)))
    fn = int(np.sum(es_ataque & (prediccion == 0)))
    vp = int(np.sum(es_ataque & (prediccion == 1)))
    apcer = fn / (fn + vp) if (fn + vp) > 0 else float("nan")
    bpcer = fp / (fp + vn) if (fp + vn) > 0 else float("nan")

    return {
        "umbral": float(umbral),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "exactitud": (vn + vp) / len(etiquetas),
        "auc": float(roc_auc_score(etiquetas, prob_ataque)) if len(np.unique(etiquetas)) == 2 else float("nan"),
        "n_reales": int(es_real.sum()),
        "n_ataques": int(es_ataque.sum()),
        "matriz_confusion": [[vn, fp], [fn, vp]],
        "apcer_por_tipo": {t: float(np.mean(prediccion[es_ataque & (tipos_ataque == t)] == 0))
                           for t in sorted(set(tipos_ataque[es_ataque]))},
    }


def agregar_por_video(predicciones):
    return (predicciones.groupby(["id_video", "etiqueta", "tipo_ataque", "id_persona", "fuente", "split"],
                                 as_index=False)["prob_ataque"].mean())
