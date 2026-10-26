"""Dataset de la réplica exacta del Fretting-Transformer (DadaGP, modelo standard).

Pipeline del paper (Fig. 1, §3.2): filtro de pistas acústicas (GM 25/26 +
palabra de guitarra en el nombre) -> limpieza -> quitar duplicados -> split ->
extracción de TODAS las notas (start, end, pitch, cuerda, traste).
Modelo "standard": solo afinación estándar sin cejilla.

Lee data/acoustic_index.json (scripts/build_acoustic_index.py) y escribe
data/paper/{train,val,test}.jsonl.gz. Imprime la comparación con el paper.

Uso:
    python3.12 scripts/build_paper_dataset.py
"""

import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chunking import max_notes_for, training_windows
from src.data.dadagp_gp import (
    dedup_paper, is_guitar_named, load_index, load_track_poly, save_tracks, split_paper,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(PROJECT), "DadaGP-v1.1")
OUT = os.path.join(PROJECT, "data", "paper")

PAPER = {"tracks": 2301, "train_seq": 16451, "val_seq": 1819}


def _load(info):
    try:
        return load_track_poly(ROOT, info)
    except Exception:
        return None


def main():
    infos, _ = load_index(os.path.join(PROJECT, "data", "acoustic_index.json"))
    named = [i for i in infos if is_guitar_named(i)]
    std = [i for i in named if i.usable]
    uniq = dedup_paper(std)
    print(f"Pistas acústicas (GM 25/26):           {len(infos)}")
    print(f"+ palabra de guitarra en el nombre:    {len(named)}")
    print(f"+ afinación estándar, sin cejilla:     {len(std)}")
    print(f"+ sin duplicados (1 fichero/título):   {len(uniq)} pistas, "
          f"{len({i.song for i in uniq})} ficheros   (paper: {PAPER['tracks']} pistas)")

    splits = split_paper(uniq)
    os.makedirs(OUT, exist_ok=True)
    max_notes = max_notes_for()
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        for name, part in splits.items():
            songs = [s for s in ex.map(_load, part, chunksize=8) if s is not None]
            save_tracks(os.path.join(OUT, f"{name}.jsonl.gz"), songs)
            n = [len(s["notes"]) for s in songs]
            seq = sum(len(training_windows(s, max_notes)) for s in songs)
            ref = f"  (paper: {PAPER[name + '_seq']})" if name + "_seq" in PAPER else ""
            print(f"{name:5s}: {len(songs)}/{len(part)} pistas (≥8 notas) | notas {sum(n)}, "
                  f"mediana {st.median(n):.0f}/pista | secuencias de 512 tokens: {seq}{ref}")
    print(f"\nTiempo: {time.time() - t0:.0f}s | salida: {OUT}")


if __name__ == "__main__":
    main()
