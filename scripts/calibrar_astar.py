"""Calibración de los pesos de A* con acordes sobre la validación del conjunto mezclado.

Búsqueda en rejilla de 81 combinaciones (w_move, w_high, w_open, w_span). Cada
combinación se puntúa con la media de la tab accuracy en las tres fuentes de la
validación (DadaGP, GuitarSet y GAPS), para que ningún repertorio domine. De DadaGP
se usan 150 pistas para acotar el tiempo de cálculo. Nunca se usa el test.

Uso:
    python3.12 scripts/calibrar_astar.py
Salida: experiments/calibracion_astar.json (mejor combinación y las cinco primeras).
"""

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks
from src.metrics import tab_accuracy
from src.models.astar import AStarChords, ChordWeights

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {
    "w_move": (0.5, 1.0, 2.0),
    "w_high": (0.1, 0.3, 0.6),
    "w_open": (-0.5, -0.2, 0.0),
    "w_span": (0.25, 0.5, 1.0),
}


def accuracy(model, songs):
    """Tab accuracy ponderada por nota."""
    n = sum(len(s["notes"]) for s in songs)
    return sum(tab_accuracy(model.predict([p for p, _ in s["notes"]], s["starts"]),
                            [t for t, _ in s["tabs"]]) * len(s["notes"]) for s in songs) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--dadagp-tracks", type=int, default=150)
    args = ap.parse_args()

    by_source = defaultdict(list)
    for s in load_tracks(os.path.join(args.data, "val.jsonl.gz")):
        by_source[s["meta"]["source"]].append(s)
    by_source["dadagp"] = by_source["dadagp"][: args.dadagp_tracks]

    results, t0 = [], time.time()
    for combo in itertools.product(*GRID.values()):
        weights = dict(zip(GRID, combo))
        model = AStarChords(weights=ChordWeights(**weights))
        acc = {src: accuracy(model, songs) for src, songs in by_source.items()}
        results.append({"media": sum(acc.values()) / len(acc), "por_fuente": acc, "pesos": weights})
    results.sort(key=lambda r: -r["media"])

    print(f"{len(results)} combinaciones en {time.time() - t0:.0f} s")
    for r in results[:5]:
        print(f"media {r['media']:.2%}  "
              + "  ".join(f"{k} {v:.2%}" for k, v in r["por_fuente"].items()) + f"  {r['pesos']}")
    os.makedirs(os.path.join(PROJECT, "experiments"), exist_ok=True)
    with open(os.path.join(PROJECT, "experiments", "calibracion_astar.json"), "w") as fh:
        json.dump({"mejor": results[0]["pesos"], "top5": results[:5]}, fh, indent=1)


if __name__ == "__main__":
    main()
