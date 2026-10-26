"""Evaluación del benchmark con las métricas y tablas del paper.

Tabla 5 (paper): tab accuracy y difficulty por método (baseline, A*, T5, GPT2, BERT).
A* = A-star-Guitar con la función de coste del código de Burlet y Fujinaga, sin
calibrar; "A* nota a nota" es el mismo coste sin agrupar acordes (ablación) y
"A* variante propia" la variante con coste propio calibrado (`AStarChords`).
Tabla 2 (paper): efecto del post-procesado (sin / ventana ±5 / +fallback).
Además: acierto por zona del mástil (según el traste de la tablatura real).

Los modelos se evalúan con el protocolo del paper (trozos de 20 notas + 1 de
contexto, decodificación voraz) salvo que se indique otro. Resultados en experiments/<nombre>.eval.json.

Uso:
    python3.12 scripts/evaluate_benchmark.py --data data/mixed --split test \\
        --run runs/mixed_t5 --run runs/mixed_gpt2 --constrained --name definitivo_test
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # para importar train

from src.data.dadagp_gp import load_tracks
from src.fretboard import Fretboard
from src.metrics import difficulty_score, pitch_accuracy, tab_accuracy
from src.models.astar import AStarChords, ChordWeights
from src.models.astar_burlet import AStarBurlet
from src.models.baseline_minfret import MinFretBaseline

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = [("0-4", 0, 4), ("5-9", 5, 9), ("10-14", 10, 14), ("15+", 15, 99)]


def score(preds, songs, fb):
    """Métricas ponderadas por nota + acierto por zona del mástil."""
    n = sum(len(s["notes"]) for s in songs)
    res = {"n_tracks": len(songs), "n_notes": n}
    res["tab_acc"] = sum(tab_accuracy(p, [t for t, _ in s["tabs"]]) * len(s["notes"])
                         for p, s in zip(preds, songs)) / n
    res["pitch_acc"] = sum(pitch_accuracy(p, [q for q, _ in s["notes"]], fb) * len(s["notes"])
                           for p, s in zip(preds, songs)) / n
    if all(x is not None for p in preds for x in p):
        res["difficulty"] = sum(difficulty_score(p) * len(s["notes"])
                                for p, s in zip(preds, songs)) / n
    zones = {}
    for name, lo, hi in ZONES:
        ok = tot = 0
        for p, s in zip(preds, songs):
            for x, (g, _) in zip(p, s["tabs"]):
                if lo <= g.fret <= hi:
                    tot += 1
                    ok += x == g
        zones[name] = {"share": tot / n, "tab_acc": ok / tot if tot else None}
    res["zones"] = zones
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJECT, "data", "mixed"))
    ap.add_argument("--split", default="test", help="nombre del fichero en --data: val, test, test_limpio...")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run", "--t5", dest="runs", action="append", default=[],
                    help="directorio de un run (best/, tokenizer.json, history.json); repetible")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=1)
    ap.add_argument("--commit", type=int, default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--astar-weights", default="{}",
                    help='JSON con pesos de la variante propia de A*, p. ej. \'{"w_high": 0.3}\'')
    ap.add_argument("--constrained", action="store_true",
                    help="añade, por cada run, la decodificación restringida por pitch (Hsu et al. 2026)")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    songs = load_tracks(os.path.join(args.data, f"{args.split}.jsonl.gz"))[: args.limit]
    fb = Fretboard()
    gt = [[t for t, _ in s["tabs"]] for s in songs]
    results = {"config": vars(args), "methods": {}}
    results["methods"]["ground_truth"] = score(gt, songs, fb)

    for name, model, chords in [
            ("baseline", MinFretBaseline(), False),
            ("A* nota a nota", AStarBurlet(fb), False),
            ("A*", AStarBurlet(fb), True),
            ("A* variante propia", AStarChords(fb, weights=ChordWeights(**json.loads(args.astar_weights))), True)]:
        t0 = time.time()
        if chords:
            preds = [model.predict([p for p, _ in s["notes"]], s.get("starts")) for s in songs]
        else:
            preds = [model.predict([p for p, _ in s["notes"]]) for s in songs]
        results["methods"][name] = {**score(preds, songs, fb), "seconds": time.time() - t0}

    for run in args.runs:
        from src.models import bert_fretting, gpt2_fretting, t5_fretting
        from src.tokenizer_v3 import V3Tokenizer
        from train import pick_device
        with open(os.path.join(run, "history.json")) as fh:
            arch_name = json.load(fh)["config"].get("arch", "t5")
        arch = {"t5": t5_fretting, "gpt2": gpt2_fretting, "bert": bert_fretting}[arch_name]
        label = arch_name.upper()
        if sum(1 for r in args.runs if os.path.basename(os.path.normpath(r)) != os.path.basename(os.path.normpath(run))) and \
                any(json.load(open(os.path.join(r, "history.json")))["config"].get("arch", "t5") == arch_name
                    for r in args.runs if r != run):
            label += f" [{os.path.basename(os.path.normpath(run))}]"   # varios runs de la misma arquitectura
        tok = V3Tokenizer.load(os.path.join(run, "tokenizer.json"))
        model = arch.load_model(os.path.join(run, "best")).to(pick_device("auto"))
        t0 = time.time()
        out = arch.predict_tracks(model, tok, [s["notes"] for s in songs], fb=fb, chunk=args.chunk,
                                  ctx=args.ctx, commit=args.commit, num_beams=args.beams)
        secs = time.time() - t0
        for stage, suffix in [("raw", " sin post-procesado"), ("window", " + ventana ±5"), ("final", "")]:
            results["methods"][label + suffix] = {**score([o[stage] for o in out], songs, fb),
                                                  "seconds": secs, "run": run}
        if args.constrained:
            t0 = time.time()
            out = arch.predict_tracks(model, tok, [s["notes"] for s in songs], fb=fb, chunk=args.chunk,
                                      ctx=args.ctx, commit=args.commit, num_beams=args.beams,
                                      constrained=True)
            # sin post-procesado: la restricción ya garantiza la nota y el alineamiento
            results["methods"][label + " restringida"] = {
                **score([o["raw"] for o in out], songs, fb), "seconds": time.time() - t0, "run": run}

    os.makedirs(os.path.join(PROJECT, "experiments"), exist_ok=True)
    path = os.path.join(PROJECT, "experiments", f"{args.name}.eval.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)

    m = results["methods"]
    print(f"\n{args.split}: {m['ground_truth']['n_tracks']} pistas, {m['ground_truth']['n_notes']} notas")
    print(f"{'Método':24s} {'Tab acc.':>9s} {'Pitch acc.':>11s} {'Difficulty':>11s}   "
          + "  ".join(f"{z:>6s}" for z, _, _ in ZONES))
    for name, r in m.items():
        d = f"{r['difficulty']:.4f}" if "difficulty" in r else "—"
        zs = "  ".join(f"{r['zones'][z]['tab_acc']:6.1%}" if r['zones'][z]['tab_acc'] is not None
                       else f"{'-':>6s}" for z, _, _ in ZONES)
        print(f"{name:24s} {r['tab_acc']:9.2%} {r['pitch_acc']:11.2%} {d:>11s}   {zs}")
    print("Reparto de notas por zona:", "  ".join(
        f"{z} {m['ground_truth']['zones'][z]['share']:.1%}" for z, _, _ in ZONES))
    print(f"Guardado en {path}")


if __name__ == "__main__":
    main()
