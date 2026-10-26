"""Dataset mezclado DadaGP + GuitarSet + GAPS con reparto 80/10/10 sin fugas.

Entrena, valida y evalúa con los tres conjuntos juntos. Cada conjunto se reparte
por la unidad que evita fugas entre train y test, y luego se combinan:
- DadaGP: se MANTIENE el reparto de la réplica (data/paper, 80/10/10 por
  fichero, estratificado por tonalidad) → su test es idéntico al anterior y los
  resultados son comparables con los de la réplica.
- GuitarSet: por PROGRESIÓN (15 progresiones → 11/2/2; los 6 guitarristas
  tocan las mismas), estratificado por estilo: val y test toman una progresión
  de 4 estilos distintos, así cada estilo conserva ≥ 2 progresiones en train
  (un reparto al azar dejó las 3 de jazz fuera de train).
- GAPS: 80/10/10 por OBRA (título normalizado). Las obras cuyo título coincide
  con una pista de DadaGP van al MISMO lado que esa pista (si coincide con varias,
  al primero de train/val/test), para que la misma obra no esté en train y test.
Duraciones de GuitarSet (interpretación real) ajustadas al vocabulario de DadaGP;
las de GAPS ya vienen de partitura.

Salidas en data/mixed/: {train,val,test}.jsonl.gz y test_{dadagp,guitarset,gaps}.jsonl.gz
(cada pista lleva meta["source"]). Uso:
    python3.12 scripts/build_mixed_dataset.py
"""

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks, save_tracks
from src.data.gaps import _ascii, load_gaps, overlapping_scores
from src.data.guitarset import load_guitarset, progression, snap_durations

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT)
OUT = os.path.join(PROJECT, "data", "mixed")
SPLITS = ("train", "val", "test")


def split_groups(keys, fracs=(0.8, 0.1, 0.1), seed=42):
    """Reparte grupos (claves) en train/val/test; val y test con al menos 1 grupo."""
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * fracs[2]))
    n_val = max(1, round(n * fracs[1]))
    out = {}
    for i, k in enumerate(keys):
        out[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def split_guitarset_by_style(songs, n_val=2, n_test=2, seed=42):
    """GuitarSet por progresión pero estratificado por estilo: val y test toman
    cada uno una progresión de estilos distintos (4 estilos diferentes en total),
    de modo que todos los estilos conservan al menos 2 progresiones en train."""
    from src.data.guitarset import STYLES
    rng = random.Random(seed)
    styles = sorted(STYLES)
    rng.shuffle(styles)
    assign = {progression(s): "train" for s in songs}
    for sp, chosen in (("test", styles[:n_test]), ("val", styles[n_test:n_test + n_val])):
        for st in chosen:
            progs = sorted({progression(s) for s in songs if s["meta"]["style"] == st})
            assign[rng.choice(progs)] = sp
    return assign


def tag(songs, source):
    return [{**s, "meta": {**s["meta"], "source": source}} for s in songs]


def main():
    # DadaGP: reparto existente
    dada = {sp: tag(load_tracks(os.path.join(PROJECT, "data", "paper", f"{sp}.jsonl.gz")), "dadagp")
            for sp in SPLITS}
    vocab = {d for s in dada["train"] for _, d in s["notes"]}

    # GuitarSet: por progresión
    gs = load_guitarset(os.path.join(ROOT, "GuitarSet-1.1"))
    gs_assign = split_guitarset_by_style(gs)
    gs_split = {sp: tag(snap_durations([s for s in gs if gs_assign[progression(s)] == sp], vocab),
                        "guitarset") for sp in SPLITS}

    # GAPS: por obra; las que coinciden con DadaGP siguen a su pista
    gaps = load_gaps(os.path.join(ROOT, "GAPS-v1"))
    ids = {s["meta"]["song"] for s in gaps}
    forced = {}
    for sp in SPLITS:
        names = [os.path.basename(str(t["meta"].get("song", ""))) for t in dada[sp]]
        for sid in overlapping_scores(os.path.join(ROOT, "GAPS-v1"), names, ids=ids):
            forced.setdefault(sid, sp)            # orden train → val → test
    import csv
    with open(os.path.join(ROOT, "GAPS-v1", "gaps_v1_metadata.csv"), encoding="utf-8") as fh:
        title = {(r.get("scorehash") or "").strip(): _ascii(r.get("title", "")).strip()
                 for r in csv.DictReader(fh)}
    work = lambda s: title.get(s["meta"]["song"]) or s["meta"]["song"]
    free = [s for s in gaps if s["meta"]["song"] not in forced]
    gaps_assign = split_groups(work(s) for s in free)
    gaps_split = {sp: [] for sp in SPLITS}
    for s in gaps:
        sp = forced.get(s["meta"]["song"]) or gaps_assign[work(s)]
        gaps_split[sp].append(s)
    gaps_split = {sp: tag(v, "gaps") for sp, v in gaps_split.items()}

    os.makedirs(OUT, exist_ok=True)
    for sp in SPLITS:
        save_tracks(os.path.join(OUT, f"{sp}.jsonl.gz"), dada[sp] + gs_split[sp] + gaps_split[sp])
    for src, part in (("dadagp", dada["test"]), ("guitarset", gs_split["test"]), ("gaps", gaps_split["test"])):
        save_tracks(os.path.join(OUT, f"test_{src}.jsonl.gz"), part)

    print(f"{'':10s} {'train':>16s} {'val':>16s} {'test':>16s}")
    for name, d in (("DadaGP", dada), ("GuitarSet", gs_split), ("GAPS", gaps_split)):
        print(f"{name:10s} " + " ".join(f"{len(d[sp]):5d} ({sum(len(s['notes']) for s in d[sp]):7d} n)"
                                        for sp in SPLITS))
    print(f"GAPS forzadas por coincidir con DadaGP: {Counter(forced.values())}")
    print("GuitarSet test:", sorted({progression(s) for s in gs_split['test']}),
          "| val:", sorted({progression(s) for s in gs_split['val']}))


if __name__ == "__main__":
    main()
