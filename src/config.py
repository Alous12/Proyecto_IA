from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_DATA = RAIZ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_RESULTS = RAIZ / "results"

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


def ruta_relativa(ruta):
    return Path(ruta).resolve().relative_to(RAIZ).as_posix()


def ruta_absoluta(ruta_rel):
    return RAIZ / ruta_rel
