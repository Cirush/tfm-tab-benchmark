"""GAPS como conjunto de test adicional para los modelos entrenados con DadaGP.

Descarga previa (sin audio, 7 MB, CC BY-NC-SA 4.0):
    https://zenodo.org/records/13962272 -> gaps_v1_no_audio.zip -> ../GAPS-v1/
Requiere data/paper_gs/train.jsonl.gz (para el control de fugas).

Salidas:
    data/gaps/test.jsonl.gz        todas las piezas en afinación estándar
    data/gaps/test_limpio.jsonl.gz sin las obras cuyo título aparece en el train
    data/gaps/excluidas.json       obras excluidas y con qué pista coinciden

Uso:
    python3.12 scripts/build_gaps.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks, save_tracks
from src.data.gaps import load_gaps, overlapping_scores

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(PROJECT), "GAPS-v1")
OUT = os.path.join(PROJECT, "data", "gaps")

if __name__ == "__main__":
    songs = load_gaps(ROOT)
    train = load_tracks(os.path.join(PROJECT, "data", "paper_gs", "train.jsonl.gz"))
    names = [os.path.basename(str(t["meta"].get("song", ""))) for t in train]
    excl = overlapping_scores(ROOT, names, ids={s["meta"]["song"] for s in songs})
    limpio = [s for s in songs if s["meta"]["song"] not in excl]
    os.makedirs(OUT, exist_ok=True)
    save_tracks(os.path.join(OUT, "test.jsonl.gz"), songs)
    save_tracks(os.path.join(OUT, "test_limpio.jsonl.gz"), limpio)
    with open(os.path.join(OUT, "excluidas.json"), "w") as fh:
        json.dump(excl, fh, indent=1, ensure_ascii=False)
    for name, part in (("completo", songs), ("limpio", limpio)):
        print(f"GAPS {name:8s}: {len(part)} piezas, {sum(len(s['notes']) for s in part)} notas")
    print(f"excluidas por posible fuga: {len(excl)} (detalle en {OUT}/excluidas.json)")
