import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src import config


def elegir_umbral(etiquetas, prob_ataque):
    etiquetas = np.asarray(etiquetas)
    if len(np.unique(etiquetas)) < 2:
        raise ValueError("Para elegir el umbral, validación debe tener reales y ataques.")
    fpr, tpr, umbrales = roc_curve(etiquetas, prob_ataque)
    apcer, bpcer = 1 - tpr, fpr
    i = int(np.argmin(np.abs(apcer - bpcer)))
    return float(min(umbrales[i], 1.0))


def calcular_metricas(etiquetas, prob_ataque, umbral, tipos_ataque=None):
    etiquetas = np.asarray(etiquetas).astype(int)
    prob_ataque = np.asarray(prob_ataque, dtype=float)
    prediccion = (prob_ataque >= umbral).astype(int)

    es_ataque = etiquetas == config.ETIQUETA_ATAQUE
    es_real = etiquetas == config.ETIQUETA_REAL

    vn = int(np.sum(es_real & (prediccion == 0)))
    fp = int(np.sum(es_real & (prediccion == 1)))
    fn = int(np.sum(es_ataque & (prediccion == 0)))
    vp = int(np.sum(es_ataque & (prediccion == 1)))

    apcer = fn / (fn + vp) if (fn + vp) > 0 else float("nan")
    bpcer = fp / (fp + vn) if (fp + vn) > 0 else float("nan")

    resultado = {
        "umbral": float(umbral),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "exactitud": (vn + vp) / len(etiquetas),
        "auc": float(roc_auc_score(etiquetas, prob_ataque))
               if len(np.unique(etiquetas)) == 2 else float("nan"),
        "n_reales": int(es_real.sum()),
        "n_ataques": int(es_ataque.sum()),
        "matriz_confusion": [[vn, fp], [fn, vp]],
    }

    if tipos_ataque is not None:
        tipos_ataque = np.asarray(tipos_ataque)
        por_tipo = {}
        for tipo in sorted(set(tipos_ataque[es_ataque])):
            del_tipo = es_ataque & (tipos_ataque == tipo)
            por_tipo[tipo] = float(np.mean(prediccion[del_tipo] == 0))
        resultado["apcer_por_tipo"] = por_tipo
        resultado["apcer_max_tipo"] = max(por_tipo.values()) if por_tipo else float("nan")
    return resultado


def agregar_por_video(predicciones):
    columnas = [c for c in ["id_video", "etiqueta", "tipo_ataque", "id_persona", "fuente", "split"]
                if c in predicciones.columns]
    return (predicciones.groupby(columnas, as_index=False)
            .agg(prob_ataque=("prob_ataque", "mean"), n_frames=("prob_ataque", "size")))


def metricas_desde_dataframe(df, umbral):
    return calcular_metricas(df["etiqueta"], df["prob_ataque"], umbral, df["tipo_ataque"])
