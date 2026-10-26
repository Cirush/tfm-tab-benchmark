"""Acierto según el tamaño del evento (nota suelta o acorde) en un test (DadaGP por defecto).

Explica de dónde sale la ventaja de los métodos que agrupan acordes:
  - qué parte de las notas está en notas sueltas, acordes de 2 notas y de 3 o más;
  - la tab accuracy de los métodos clásicos en cada tipo de evento;
  - cuántas digitaciones tiene un acorde de 3 o más notas (una nota por cuerda) y
    cuántas caben en la mano (como mucho 3 trastes entre las notas pisadas);
  - qué parte de esos acordes mezcla cuerdas al aire y notas pisadas en la
    tablatura real y en la de cada A*.

Los acordes se agrupan por instante de inicio, como en A*.

Uso: python3.12 scripts/analisis_acordes.py --data data/mixed [--split test_gaps]
Salida: experiments/definitivo_analisis_acordes_<test>.json (p. ej. _dadagp)
"""
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks
from src.fretboard import Fretboard
from src.models.astar import AStarChords
from src.models.astar_burlet import AStarBurlet
from src.models.baseline_minfret import MinFretBaseline

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_COMBOS = 50_000  # acordes con más combinaciones no se enumeran (muy pocos)


def tipo(n):
    return "nota suelta" if n == 1 else ("acorde de 2 notas" if n == 2 else "acorde de 3+ notas")


def digitaciones(fb, pitches):
    """(nº de digitaciones con cuerdas distintas, nº de ellas con apertura ≤ 3), o None."""
    cands = [fb.candidates(p) for p in pitches]
    if np.prod([len(c) for c in cands]) > MAX_COMBOS:
        return None
    total = tocables = 0
    for c in itertools.product(*cands):
        if len({p.string for p in c}) < len(c):
            continue
        total += 1
        pisados = [p.fret for p in c if p.fret > 0]
        tocables += not pisados or max(pisados) - min(pisados) <= 3
    return total, tocables


def mezcla_aire(frets):
    return 0 in frets and max(frets) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--split", default="test_dadagp")
    args = ap.parse_args()

    songs = load_tracks(os.path.join(args.data, f"{args.split}.jsonl.gz"))
    fb = Fretboard()
    metodos = {"baseline": (MinFretBaseline(), False), "A* nota a nota": (AStarBurlet(fb), False),
               "A*": (AStarBurlet(fb), True), "A* variante propia": (AStarChords(fb), True)}
    notas, aciertos = defaultdict(int), defaultdict(lambda: defaultdict(int))
    n_dig, n_toc, real_toc, n_acordes = [], [], 0, 0
    aire = defaultdict(int)

    for s in songs:
        p = [x for x, _ in s["notes"]]
        gt = [t for t, _ in s["tabs"]]
        preds = {m: (mod.predict(p, s["starts"]) if acordes else mod.predict(p))
                 for m, (mod, acordes) in metodos.items()}
        for g in AStarChords.groups(AStarChords(fb), s["starts"]):
            t = tipo(len(g))
            notas[t] += len(g)
            for m, pr in preds.items():
                aciertos[t][m] += sum(pr[i] == gt[i] for i in g)
            if len(g) < 3:
                continue
            n_acordes += 1
            d = digitaciones(fb, [p[i] for i in g])
            if d is not None:
                n_dig.append(d[0])
                n_toc.append(d[1])
            pisados = [gt[i].fret for i in g if gt[i].fret > 0]
            real_toc += not pisados or max(pisados) - min(pisados) <= 3
            aire["tablatura real"] += mezcla_aire([gt[i].fret for i in g])
            for m in ("A*", "A* variante propia"):
                aire[m] += mezcla_aire([preds[m][i].fret for i in g])

    total = sum(notas.values())
    n_dig, n_toc = np.array(n_dig), np.array(n_toc)
    res = {
        "por_tipo": {t: {"frac_notas": notas[t] / total,
                         "tab_acc": {m: aciertos[t][m] / notas[t] for m in metodos}}
                     for t in ("nota suelta", "acorde de 2 notas", "acorde de 3+ notas")},
        "acordes_3_mas": {
            "n_acordes": n_acordes,
            "digitaciones_media": float(n_dig.mean()), "digitaciones_mediana": float(np.median(n_dig)),
            "tocables_media": float(n_toc.mean()), "tocables_mediana": float(np.median(n_toc)),
            "frac_una_tocable": float((n_toc == 1).mean()),
            "frac_reales_apertura_3": real_toc / n_acordes,
            "frac_mezcla_aire_y_pisadas": {k: v / n_acordes for k, v in aire.items()},
        },
    }
    path = os.path.join(PROJECT, "experiments",
                        f"definitivo_analisis_acordes_{args.split.removeprefix('test_')}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("Guardado en", path)


if __name__ == "__main__":
    main()
