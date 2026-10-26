"""GuitarSet para el benchmark: conjunto completo, split train/test y dataset
combinado DadaGP (réplica del paper) + GuitarSet.

Descarga previa (solo anotaciones, 39 MB, CC BY 4.0):
    https://zenodo.org/records/3371780/files/annotation.zip -> ../GuitarSet-1.1/
Requiere data/paper/ (scripts/build_paper_dataset.py).

Salidas:
    data/guitarset/all.jsonl.gz     los 360 fragmentos
    data/guitarset/train.jsonl.gz   2 progresiones por estilo (240 fragmentos)
    data/guitarset/test.jsonl.gz    1 progresión por estilo (120), música no vista
    data/paper_gs/{train,val,test}  train = DadaGP train + GuitarSet train
                                    (duraciones ajustadas al vocabulario de DadaGP);
                                    val/test = los de DadaGP

Uso:
    python3.12 scripts/build_guitarset.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks, save_tracks
from src.data.guitarset import load_guitarset, progression, snap_durations, split_guitarset

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(PROJECT), "GuitarSet-1.1")
DATA = os.path.join(PROJECT, "data")

if __name__ == "__main__":
    songs = load_guitarset(ROOT)
    train, test = split_guitarset(songs)
    out = os.path.join(DATA, "guitarset")
    os.makedirs(out, exist_ok=True)
    for name, part in (("all", songs), ("train", train), ("test", test)):
        save_tracks(os.path.join(out, f"{name}.jsonl.gz"), part)
        print(f"GuitarSet {name:5s}: {len(part)} fragmentos, {sum(len(s['notes']) for s in part)} notas")
    print("progresiones de test:", sorted({progression(s) for s in test}))

    dada_train = load_tracks(os.path.join(DATA, "paper", "train.jsonl.gz"))
    vocab = {d for s in dada_train for _, d in s["notes"]}
    comb = os.path.join(DATA, "paper_gs")
    os.makedirs(comb, exist_ok=True)
    save_tracks(os.path.join(comb, "train.jsonl.gz"), dada_train + snap_durations(train, vocab))
    for split in ("val", "test"):
        shutil.copy(os.path.join(DATA, "paper", f"{split}.jsonl.gz"), os.path.join(comb, f"{split}.jsonl.gz"))
    print(f"paper_gs train: {len(dada_train)} pistas DadaGP + {len(train)} fragmentos GuitarSet")
