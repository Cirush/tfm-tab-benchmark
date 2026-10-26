"""Simulación que justifica la variante de post-procesado elegida.

Toma pistas reales de val, las trocea en chunks de 20 notas (como la
inferencia del paper), les inyecta errores tipo modelo (notas saltadas, TABs
insertados, TABs cambiados) y compara tres variantes de la ventana ±5:
  naive   : centrada en i (lectura literal del paper)  <- la elegida
  drift   : centrada en i + desfase del último match
  hibrido : centrada en i, desempate hacia el lado del desfase
Todas terminan con el mismo fallback. Usa data/mixed/val.jsonl.gz.

Con 300 pistas de validación (~310k notas), naive supera a drift en los 4
escenarios (drift pierde 0,3-0,5 puntos) e hibrido ≈ naive (±0,15).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dadagp_gp import load_tracks
from src.fretboard import Fretboard, Position
from src.metrics import tab_accuracy
from src.postprocess import PAPER_WINDOW, fill_fallback

FB = Fretboard()
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mixed")
SCENARIOS = [  # (saltadas, insertadas, cambiadas)
    (0.02, 0.02, 0.03), (0.05, 0.05, 0.10), (0.10, 0.0, 0.25), (0.0, 0.10, 0.25),
]


def _ok(t, pitch):
    return 1 <= t.string <= 6 and t.fret <= 24 and FB.pitch_of(t) == pitch


def match(pred, pitches, mode, w=PAPER_WINDOW):
    out, shift = [], 0
    for i, pitch in enumerate(pitches):
        side = (shift > 0) - (shift < 0)
        if mode == "hibrido":
            offs = sorted(range(-w, w + 1), key=lambda d: (abs(d), d != side, d))
        else:
            offs = sorted(range(-w, w + 1), key=lambda d: (abs(d), d))
        center = i + shift if mode == "drift" else i
        m = None
        for d in offs:
            j = center + d
            if 0 <= j < len(pred) and _ok(pred[j], pitch):
                m, shift = pred[j], j - i
                break
        out.append(m)
    return fill_fallback(out, pitches, FB)


def corrupt(gt, rng, drop, ins, sub):
    pred = []
    for t in gt:
        r = rng.random()
        if r < drop:
            continue
        if r < drop + sub:
            t = Position(rng.randint(1, 6), rng.randint(0, 15))
        pred.append(t)
        if rng.random() < ins:
            pred.append(Position(rng.randint(1, 6), rng.randint(0, 15)))
    return pred


def main(n_tracks=300):
    songs = load_tracks(os.path.join(DATA, "val.jsonl.gz"))[:n_tracks]
    for drop, ins, sub in SCENARIOS:
        rng = random.Random(0)
        tot = {m: 0.0 for m in ("naive", "drift", "hibrido")}
        n = 0
        for s in songs:
            for k in range(0, len(s["notes"]), 20):
                gt = [t for t, _ in s["tabs"][k:k + 20]]
                pitches = [p for p, _ in s["notes"][k:k + 20]]
                pred = corrupt(gt, rng, drop, ins, sub)
                for m in tot:
                    tot[m] += tab_accuracy(match(pred, pitches, m), gt) * len(gt)
                n += len(gt)
        print(f"saltadas {drop:.0%} insertadas {ins:.0%} cambiadas {sub:.0%}: "
              + "  ".join(f"{m} {v / n:.2%}" for m, v in tot.items()))


if __name__ == "__main__":
    main()
