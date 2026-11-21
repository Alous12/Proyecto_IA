import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_DATA = RAIZ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"

DIR_LCC = DIR_RAW / "lcc_fasd" / "LCC_FASD"
DIR_PROPIO = DIR_RAW / "propio"

DIR_FRAMES = DIR_PROCESSED / "frames"
DIR_ROSTROS = DIR_PROCESSED / "rostros"
DIR_MATRICES = DIR_PROCESSED / "matrices"

MANIFEST_FRAMES = DIR_PROCESSED / "manifest_frames.csv"
MANIFEST_ROSTROS = DIR_PROCESSED / "manifest_rostros.csv"
MANIFEST = DIR_PROCESSED / "manifest.csv"

COLUMNAS_MANIFEST = ["ruta_frame", "etiqueta", "tipo_ataque", "id_persona",
                     "id_video", "fuente", "split"]

ETIQUETA_REAL = 0
ETIQUETA_ATAQUE = 1

TIPOS_PROPIO = {
    "real": (ETIQUETA_REAL, "ninguno"),
    "impreso": (ETIQUETA_ATAQUE, "impreso"),
    "pantalla": (ETIQUETA_ATAQUE, "pantalla"),
}
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

FUENTES = ["lcc", "lcc_eval", "propio"]
SPLITS = ["train", "val", "test"]

SEMILLA = 42

FRAMES_POR_VIDEO = 10

MARGEN_ROSTRO = 0.25
TAMANO_ROSTRO = 224
TAMANO_MIN_ROSTRO = 0.15

TAMANO_REGRESION = 64

PROPORCION_SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def definir_results(base):
    """Único lugar donde se arman las rutas de results/ (los tests la llaman con otra base)."""
    global DIR_RESULTS, DIR_CORRIDAS, DIR_FINALES, DIR_RESUMEN, DIR_FIGURAS_GRUPOS
    global RUTA_EXPERIMENTOS_CSV, RUTA_EXPERIMENTOS_MD, RUTA_MEJORES_MODELOS
    global RUTA_VERIFICACION_GRADIENTE, RUTA_COMPARACION_CSV, RUTA_COMPARACION_MD
    DIR_RESULTS = Path(base)
    DIR_CORRIDAS = DIR_RESULTS / "experimentos"
    DIR_FINALES = DIR_RESULTS / "finales"
    DIR_RESUMEN = DIR_RESULTS / "resumen"
    DIR_FIGURAS_GRUPOS = DIR_RESUMEN / "figuras"
    RUTA_EXPERIMENTOS_CSV = DIR_RESUMEN / "experimentos.csv"
    RUTA_EXPERIMENTOS_MD = DIR_RESUMEN / "experimentos.md"
    RUTA_MEJORES_MODELOS = DIR_RESUMEN / "mejores_modelos.json"
    RUTA_VERIFICACION_GRADIENTE = DIR_RESUMEN / "verificacion_gradiente.txt"
    RUTA_COMPARACION_CSV = DIR_RESUMEN / "comparacion_modelos.csv"
    RUTA_COMPARACION_MD = DIR_RESUMEN / "comparacion_modelos.md"


definir_results(RAIZ / "results")


def ruta_corrida(nombre, archivo=None):
    """Carpeta de una corrida en results/experimentos/, o un archivo dentro de ella."""
    return DIR_CORRIDAS / nombre / (archivo or "")


def ruta_final(modelo):
    return DIR_FINALES / modelo


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def guardar_json(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)


def ruta_relativa(ruta):
    return Path(ruta).resolve().relative_to(RAIZ).as_posix()


def ruta_absoluta(ruta_rel):
    return RAIZ / ruta_rel
